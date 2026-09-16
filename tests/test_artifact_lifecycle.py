"""Run-owned files: aliases, cancellation, failed writes and atomic publication."""
import numpy as np
import pytest
from ArrNorm.core import raster_io as rio
from ArrNorm.core.arrnorm import Normalization
from ArrNorm.tests.helpers import Feedback, build_pair, make_normalization, read_raster
from osgeo import gdal

try:
    from qgis.core import QgsProcessingException
except ImportError:
    QgsProcessingException = Exception


class TestCancelCleanup:
    def test_cancel_during_radcal_leaves_no_intermediates(self, tmp_path):
        # Offset reference forces the clip step, so there is something to leak.
        build_pair(tmp_path, ref_offset=(37.0, 11.0), seed=3)
        norm = make_normalization(tmp_path)
        norm.feedback._cancel_when = 'Radcal process'
        norm.run()

        assert not (tmp_path / 'out.tif').exists()
        leftover = sorted(p.name for p in tmp_path.iterdir())
        assert leftover == ['ref.tif', 'tgt.tif'], leftover

    def test_error_in_imad_leaves_no_intermediates(self, tmp_path):
        def _zero_band1(ref):
            ref[0] = np.zeros(ref[0].shape)

        build_pair(tmp_path, ref_offset=(21.0, 7.0), seed=3, mutate_ref=_zero_band1)
        norm = make_normalization(tmp_path)
        with pytest.raises(QgsProcessingException):
            norm.run()

        leftover = sorted(p.name for p in tmp_path.iterdir())
        assert leftover == ['ref.tif', 'tgt.tif'], leftover


@pytest.mark.parametrize('name', ['tgt_radcal.tif', 'tgt_norm_masked.tif',
                                 'MAD(ref&tgt.tif).tif', 'ref_tgt_clip.tif'])
def test_output_named_like_an_intermediate_survives(tmp_path, name):
    build_pair(tmp_path)
    norm = make_normalization(tmp_path, nodata_mask=True)
    norm.output_file = str(tmp_path / name)
    norm.run()
    assert read_raster(norm.output_file).max() > 0
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(['ref.tif', 'tgt.tif', name])


@pytest.mark.parametrize('name', ['tgt_radcal.tif', 'tgt_norm_masked.tif', 'ref (1)&copy.tif'])
def test_input_names_cannot_be_overwritten_or_parsed(tmp_path, name):
    build_pair(tmp_path)
    ref = tmp_path / name
    (tmp_path / 'ref.tif').rename(ref)
    original = ref.read_bytes()
    norm = Normalization(str(ref), str(tmp_path / 'tgt.tif'), 8, .99, .95,
                         False, False, None, True, None, False, str(tmp_path / 'out.tif'), Feedback())
    norm.run()
    assert ref.read_bytes() == original
    assert read_raster(norm.output_file).max() > 0


@pytest.mark.parametrize('alias', ['same', 'symlink', 'hardlink'])
def test_destination_alias_of_input_rejected(tmp_path, alias):
    build_pair(tmp_path)
    output = tmp_path / 'ref.tif'
    original = output.read_bytes()
    if alias != 'same':
        output = tmp_path / 'alias.tif'
        if alias == 'symlink':
            output.symlink_to(tmp_path / 'ref.tif')
        else:
            output.hardlink_to(tmp_path / 'ref.tif')
    with pytest.raises(QgsProcessingException, match='overwrite an input'):
        Normalization(str(tmp_path / 'ref.tif'), str(tmp_path / 'tgt.tif'), 8, .99, .95,
                      False, False, None, False, None, False, str(output), Feedback())
    assert (tmp_path / 'ref.tif').read_bytes() == original


@pytest.mark.parametrize('stage', ['iMad process', 'Radcal process', 'Making nodata mask',
                                  'Applying nodata mask'])
def test_cancel_preserves_existing_output_and_cleans_workspace(tmp_path, stage):
    build_pair(tmp_path, ref_offset=(37, 11))
    output = tmp_path / 'out.tif'
    output.write_bytes(b'previous output stays intact')
    norm = make_normalization(tmp_path, nodata_mask=True, keep_mask_layer=True)
    norm.feedback._cancel_when = stage
    assert norm.run() is None
    assert output.read_bytes() == b'previous output stays intact'
    assert sorted(p.name for p in tmp_path.iterdir()) == ['out.tif', 'ref.tif', 'tgt.tif']
    assert not any('DONE' in msg for msg in norm.feedback.messages)


@pytest.mark.parametrize('method,mode', [('WriteArray', 'raise'), ('WriteArray', 'status'),
                                      ('FlushCache', 'raise'), ('FlushCache', 'status')])
def test_write_failure_keeps_destination_and_closes_handles(tmp_path, monkeypatch, method, mode):
    build_pair(tmp_path)
    destination = tmp_path / 'out.tif'
    destination.write_bytes(b'previous output')
    handles = []
    create = rio.create_raster

    def record(*args, **kwargs):
        handle = create(*args, **kwargs)
        handles.append(handle)
        return handle

    def fail(*args, **kwargs):
        if mode == 'raise':
            raise RuntimeError('injected disk write error')
        return gdal.CE_Failure

    monkeypatch.setattr(rio, 'create_raster', record)
    monkeypatch.setattr(gdal.Band if method == 'WriteArray' else gdal.Dataset, method, fail)
    norm = make_normalization(tmp_path)
    with pytest.raises(RuntimeError) as caught:
        norm.run()
    assert caught.traceback  # handles must close even while the traceback is retained
    assert handles and all(handle._dataset is None for handle in handles)
    assert destination.read_bytes() == b'previous output'
    assert sorted(p.name for p in tmp_path.iterdir()) == ['out.tif', 'ref.tif', 'tgt.tif']


def test_nested_runs_have_isolated_workspaces(tmp_path):
    build_pair(tmp_path)
    first, second = make_normalization(tmp_path), make_normalization(tmp_path)
    second.output_file = str(tmp_path / 'other.tif')
    push = first.feedback.pushInfo

    def interleave(msg):
        push(msg)
        if 'Radcal process' in msg:
            second.run()
    first.feedback.pushInfo = interleave
    first.run()
    np.testing.assert_array_equal(read_raster(first.output_file), read_raster(second.output_file))
    assert not list(tmp_path.glob('.arrnorm-*'))


@pytest.mark.parametrize('cancel', [True, False])
def test_publication_failure_restores_previous_mask(tmp_path, monkeypatch, cancel):
    from ArrNorm.core import artifacts
    build_pair(tmp_path)
    norm = make_normalization(tmp_path, nodata_mask=True, keep_mask_layer=True)
    output, mask = tmp_path / 'out.tif', tmp_path / 'out_Mask.tif'
    output.write_bytes(b'original output')
    mask.write_bytes(b'original mask')
    replace = artifacts.os.replace

    def intercept(source, destination):
        if str(destination) == str(output) and not cancel:
            raise OSError('injected publication failure')
        replace(source, destination)
        if str(destination) == str(mask) and cancel:
            norm.feedback._canceled = True

    monkeypatch.setattr(artifacts.os, 'replace', intercept)
    if cancel:
        assert norm.run() is None
    else:
        with pytest.raises(OSError, match='publication failure'):
            norm.run()
    assert output.read_bytes() == b'original output'
    assert mask.read_bytes() == b'original mask'
    assert not list(tmp_path.glob('.arrnorm-*'))


def test_cancellation_during_actual_mask_writes_prevents_publication(tmp_path, monkeypatch):
    build_pair(tmp_path)
    norm = make_normalization(tmp_path, nodata_mask=True)
    output = tmp_path / 'out.tif'
    output.write_bytes(b'previous output')
    write = gdal.Band.WriteArray

    def cancel_after_mask_write(band, *args, **kwargs):
        result = write(band, *args, **kwargs)
        if band.DataType == gdal.GDT_Byte:
            norm.feedback._canceled = True
        return result

    monkeypatch.setattr(gdal.Band, 'WriteArray', cancel_after_mask_write)
    assert norm.run() is None
    assert output.read_bytes() == b'previous output'
    assert not list(tmp_path.glob('.arrnorm-*'))


def test_vrt_destination_must_not_overwrite_a_backing_input(tmp_path):
    build_pair(tmp_path)
    original = (tmp_path / 'ref.tif').read_bytes()
    with rio.Raster(gdal.Translate(str(tmp_path / 'ref.vrt'), str(tmp_path / 'ref.tif'),
                                   format='VRT'), writable=True):
        pass
    with pytest.raises(QgsProcessingException, match='overwrite an input'):
        Normalization(str(tmp_path / 'ref.vrt'), str(tmp_path / 'tgt.tif'), 8, .99, .95,
                      False, False, None, False, None, False,
                      str(tmp_path / 'ref.tif'), Feedback())
    assert (tmp_path / 'ref.tif').read_bytes() == original


def test_missing_destination_directory_rejected_at_construction(tmp_path):
    build_pair(tmp_path)
    destination = tmp_path / 'missing' / 'out.tif'
    with pytest.raises(QgsProcessingException, match='[Dd]irectory'):
        Normalization(str(tmp_path / 'ref.tif'), str(tmp_path / 'tgt.tif'), 8, .99, .95,
                      False, False, None, False, None, False, str(destination), Feedback())
    assert not destination.parent.exists()


def test_destination_directory_removed_before_run_has_processing_error(tmp_path):
    build_pair(tmp_path)
    destination_dir = tmp_path / 'output'
    destination_dir.mkdir()
    norm = Normalization(str(tmp_path / 'ref.tif'), str(tmp_path / 'tgt.tif'), 8, .99, .95,
                         False, False, None, False, None, False,
                         str(destination_dir / 'out.tif'), Feedback())
    destination_dir.rmdir()
    with pytest.raises(QgsProcessingException, match='[Dd]irectory'):
        norm.run()
    assert not destination_dir.exists()


@pytest.mark.parametrize('error', [FileNotFoundError, PermissionError])
def test_workspace_creation_error_is_explained_and_chained(tmp_path, monkeypatch, error):
    from ArrNorm.core import artifacts
    build_pair(tmp_path)
    norm = make_normalization(tmp_path)
    failure = error('injected workspace failure')

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(artifacts.tempfile, 'TemporaryDirectory', fail)
    with pytest.raises(QgsProcessingException, match='workspace|directory') as caught:
        norm.run()
    causes = []
    cause = caught.value.__cause__
    while cause is not None:
        causes.append(cause)
        cause = cause.__cause__
    assert failure in causes
    assert sorted(p.name for p in tmp_path.iterdir()) == ['ref.tif', 'tgt.tif']
