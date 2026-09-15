"""Run-owned intermediates and atomic, single-file GeoTIFF publication."""
import os
import tempfile

from .raster_io import check_cancel


class ArtifactError(ValueError):
    """Invalid destination or unavailable workspace, with a user-facing message."""


def same_path(first, second):
    if os.path.normcase(os.path.realpath(first)) == os.path.normcase(os.path.realpath(second)):
        return True
    return os.path.exists(first) and os.path.exists(second) and os.path.samefile(first, second)


class RunArtifacts:
    def __init__(self, output, inputs, feedback, keep_mask=False):
        self.output = os.path.abspath(output)
        self.mask_output = os.path.splitext(self.output)[0] + '_Mask.tif'
        self.inputs = tuple(inputs)
        self.feedback = feedback
        self.keep_mask = keep_mask
        self._temp = None
        self.validate()

    def validate(self):
        if os.path.splitext(self.output)[1].lower() not in ('.tif', '.tiff'):
            raise ArtifactError('Normalized output must be a GeoTIFF (.tif or .tiff).')
        parent = os.path.dirname(self.output)
        if not os.path.isdir(parent):
            raise ArtifactError(f'Output directory does not exist or is not a directory: {parent}')
        destinations = [self.output] + ([self.mask_output] if self.keep_mask else [])
        for destination in destinations:
            if any(same_path(destination, source) for source in self.inputs):
                raise ArtifactError(f'Output must not overwrite an input raster: {destination}')

    def path(self, name):
        if self._temp is None:
            try:
                self._temp = tempfile.TemporaryDirectory(prefix='.arrnorm-',
                                                         dir=os.path.dirname(self.output))
            except OSError as exc:
                raise ArtifactError(f'Cannot create a temporary workspace in output directory '
                                    f'{os.path.dirname(self.output)}: {exc}') from exc
        return os.path.join(self._temp.name, name)

    def publish(self, image, mask=None):
        self.validate()  # repeat after processing, in case a destination alias changed
        check_cancel(self.feedback)
        backup = None
        mask_published = False
        try:
            if self.keep_mask and mask is not None:
                if os.path.exists(self.mask_output):
                    backup = self.path('previous_mask.tif')
                    os.replace(self.mask_output, backup)
                os.replace(mask, self.mask_output)
                mask_published = True
            check_cancel(self.feedback)
            os.replace(image, self.output)
        except BaseException:
            # The main raster is the commit point. Restore the auxiliary mask
            # if cancellation or a failed replace prevents that commit.
            if backup is not None:
                os.replace(backup, self.mask_output)
            elif mask_published:
                os.remove(self.mask_output)
            raise

    def clean(self):
        if self._temp is not None:
            try:
                self._temp.cleanup()
            except OSError as exc:
                # Keep the original processing failure; report a cleanup problem.
                if self.feedback is not None:
                    self.feedback.reportError(f'Cannot remove temporary workspace: {exc}')
            else:
                self._temp = None
