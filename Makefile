#/***************************************************************************
# ArrNorm
#
#  Automatic relative radiometric normalization
#							 -------------------
#		copyright			: (C) 2026 by Xavier Corredor Llano, SMByC
#		email				: xcorredorl@ideam.gov.co
# ***************************************************************************/
#
#/***************************************************************************
# *																		 *
# *   This program is free software; you can redistribute it and/or modify  *
# *   it under the terms of the GNU General Public License as published by  *
# *   the Free Software Foundation; either version 2 of the License, or	 *
# *   (at your option) any later version.								   *
# *																		 *
# ***************************************************************************/

#################################################
# Edit the following to match your sources lists
#################################################


#Add iso code for any locales you want to support here (space separated)
# default is no locales
# LOCALES = af
LOCALES =

# If locales are enabled, set the name of the lrelease binary on your system. If
# you have trouble compiling the translations, you may have to specify the full path to
# lrelease
#LRELEASE = lrelease
#LRELEASE = lrelease-qt4


# translation
SOURCES = \
	__init__.py \
	ArrNorm_algorithm.py \
	ArrNorm_plugin.py \
	ArrNorm_provider.py

PLUGINNAME = ArrNorm

PY_FILES = \
	__init__.py \
	ArrNorm_algorithm.py \
	ArrNorm_plugin.py \
	ArrNorm_provider.py

UI_FILES =

EXTRAS = metadata.txt LICENSE

EXTRA_DIRS = core icons gui

COMPILED_RESOURCE_FILES = resources.py

PEP8EXCLUDE=pydev,resources.py,conf.py,third_party,ui

# Install paths. Defaults target QGIS 4; override for QGIS 3 builds, e.g.:
#   make deploy QGISDIR=.local/share/QGIS/QGIS3/profiles/default
QGISDIR?=.local/share/QGIS/QGIS4/profiles/default

#################################################
# Normally you would not need to edit below here
#################################################

RESOURCE_SRC=$(shell grep '^ *<file' resources.qrc | sed 's@</file>@@g;s/.*>//g' | tr '\n' ' ')

HELP = README.md

PLUGIN_UPLOAD = python3 plugin_upload.py -u xaviercll

default: compile

compile: $(COMPILED_RESOURCE_FILES)

# Supports Qt5 (pyrcc5 / QGIS 3.x) and Qt6 (pyside6-rcc / QGIS 4.x).
# After compilation, the PyQt5 import is rewritten to the qgis.PyQt shim so
# the same resources.py file loads correctly under both QGIS versions.
%.py : %.qrc $(RESOURCE_SRC)
	@if command -v pyside6-rcc > /dev/null 2>&1; then \
		echo "Using pyside6-rcc (Qt6/QGIS 4.x)"; \
		pyside6-rcc -o $*.py $<; \
	elif command -v pyrcc5 > /dev/null 2>&1; then \
		echo "Using pyrcc5 (Qt5/QGIS 3.x)"; \
		pyrcc5 -o $*.py $<; \
	else \
		echo "Error: Neither pyside6-rcc nor pyrcc5 found."; \
		echo "Install pyside6 (Qt6/QGIS 4.x) or pyrcc5 (Qt5/QGIS 3.x)."; \
		exit 1; \
	fi
	sed -i 's/^from PyQt5 import QtCore/from qgis.PyQt import QtCore/' $*.py
	sed -i 's/^from PySide6 import QtCore/from qgis.PyQt import QtCore/' $*.py

%.qm : %.ts
	$(LRELEASE) $<

test:
	@echo
	@echo "----------------------"
	@echo "Regression Test Suite"
	@echo "----------------------"

	@export PYTHONPATH=`pwd`/..:$(PYTHONPATH); \
		export QGIS_DEBUG=0; \
		export QGIS_LOG_FILE=/dev/null; \
		export QT_QPA_PLATFORM=$${QT_QPA_PLATFORM:-offscreen}; \
		python3 -m pytest tests/ -v
	@echo "----------------------"
	@echo "Real QGIS/Qt tests run when QGIS 3.36+ Python bindings are available."
	@echo "Set QGIS_PREFIX_PATH if your QGIS installation uses a custom prefix."
	@echo "----------------------"

deploy: compile doc transcompile
	@echo
	@echo "------------------------------------------"
	@echo "Deploying plugin to your QGIS 4 directory."
	@echo "------------------------------------------"
	# The deploy  target only works on unix like operating system where
	# the Python plugin directory is located at:
	# $HOME/$(QGISDIR)/python/plugins
	mkdir -p $(HOME)/$(QGISDIR)/python/plugins/$(PLUGINNAME)
	cp -vf $(PY_FILES) $(COMPILED_RESOURCE_FILES) $(HOME)/$(QGISDIR)/python/plugins/$(PLUGINNAME)
	#cp -vf $(UI_FILES) $(HOME)/$(QGISDIR)/python/plugins/$(PLUGINNAME)
	cp -vf $(EXTRAS) $(HOME)/$(QGISDIR)/python/plugins/$(PLUGINNAME)
	#cp -vfr i18n $(HOME)/$(QGISDIR)/python/plugins/$(PLUGINNAME)
	cp -vfr $(HELP) $(HOME)/$(QGISDIR)/python/plugins/$(PLUGINNAME)/help
	# Copy extra directories if any
	cp -vfr $(EXTRA_DIRS) $(HOME)/$(QGISDIR)/python/plugins/$(PLUGINNAME)


# The dclean target removes compiled python files from plugin directory
# also deletes any .git entry
dclean:
	@echo
	@echo "-----------------------------------"
	@echo "Removing any compiled python files."
	@echo "-----------------------------------"
	find $(HOME)/$(QGISDIR)/python/plugins/$(PLUGINNAME) -iname "*.pyc" -delete
	find $(HOME)/$(QGISDIR)/python/plugins/$(PLUGINNAME) -iname ".git" -prune -exec rm -Rf {} \;
	find $(HOME)/$(QGISDIR)/python/plugins/$(PLUGINNAME) -iname "__pycache__" -prune -exec rm -Rf {} \;


derase:
	@echo
	@echo "-------------------------"
	@echo "Removing deployed plugin."
	@echo "-------------------------"
	rm -Rf $(HOME)/$(QGISDIR)/python/plugins/$(PLUGINNAME)

zip: compile
	@echo
	@echo "---------------------------"
	@echo "Creating plugin zip bundle."
	@echo "---------------------------"
	rm -f $(PLUGINNAME).zip
	mkdir -p .pkg_tmp/$(PLUGINNAME)
	cp -f $(PY_FILES) $(COMPILED_RESOURCE_FILES) $(EXTRAS) .pkg_tmp/$(PLUGINNAME)/
	@for d in $(EXTRA_DIRS); do \
		if [ -d "$$d" ]; then cp -rf $$d .pkg_tmp/$(PLUGINNAME)/; fi; \
	done
	find .pkg_tmp -type d \( -name "__pycache__" -o -name "*.dist-info" -o -name "*.egg-info" \) -prune -exec rm -rf {} \;
	find .pkg_tmp -type f \( -name "*.pyc" -o -name "*.pyo" -o -name "*.sh" -o -name "*.db" -o -name "AGENTS.md" \) -delete
	cd .pkg_tmp && zip -9r ../$(PLUGINNAME).zip $(PLUGINNAME)
	rm -rf .pkg_tmp
	@echo "Created package: $(PLUGINNAME).zip"

package: compile
	# Create a zip package of the plugin named $(PLUGINNAME).zip.
	# This requires use of git (your plugin development directory must be a
	# git repository).
	# To use, pass a valid commit or tag as follows:
	#   make package VERSION=Version_0.3.2
	@echo
	@echo "------------------------------------"
	@echo "Exporting plugin to zip package.    "
	@echo "------------------------------------"
	rm -f $(PLUGINNAME).zip
	git archive --prefix=$(PLUGINNAME)/ -o $(PLUGINNAME).zip $(VERSION)
	echo "Created package: $(PLUGINNAME).zip"

upload: zip
	@echo
	@echo "-------------------------------------"
	@echo "Uploading plugin to QGIS Plugin repo."
	@echo "-------------------------------------"
	$(PLUGIN_UPLOAD) $(PLUGINNAME).zip

transup:
	@echo
	@echo "------------------------------------------------"
	@echo "Updating translation files with any new strings."
	@echo "------------------------------------------------"
	@chmod +x scripts/update-strings.sh
	@scripts/update-strings.sh $(LOCALES)

transcompile:
	@echo
	@echo "----------------------------------------"
	@echo "Compiled translation files to .qm files."
	@echo "----------------------------------------"
	#@chmod +x scripts/compile-strings.sh
	#@scripts/compile-strings.sh $(LRELEASE) $(LOCALES)

transclean:
	@echo
	@echo "------------------------------------"
	@echo "Removing compiled translation files."
	@echo "------------------------------------"
	rm -f i18n/*.qm

clean:
	@echo
	@echo "------------------------------------"
	@echo "Removing generated files"
	@echo "------------------------------------"
	rm -f $(COMPILED_RESOURCE_FILES)
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

doc:
	@echo
	@echo "------------------------------------"
	@echo "Building documentation using sphinx."
	@echo "------------------------------------"
	# cd help; make html

pylint:
	@echo
	@echo "-----------------"
	@echo "Pylint violations"
	@echo "-----------------"
	@pylint --reports=n --rcfile=pylintrc . || true
	@echo
	@echo "----------------------"
	@echo "If you get a 'no module named qgis.core' error, try sourcing"
	@echo "the helper script we have provided first then run make pylint."
	@echo "e.g. source run-env-linux.sh <path to qgis install>; make pylint"
	@echo "----------------------"

# Run PEP8 style checking (pycodestyle is the successor of the retired
# 'pep8' tool, renamed in 2016)
#https://pypi.python.org/pypi/pycodestyle
pep8: pycodestyle

pycodestyle:
	@echo
	@echo "-----------"
	@echo "PEP8 issues"
	@echo "-----------"
	@pycodestyle --repeat --ignore=E203,E121,E122,E123,E124,E125,E126,E127,E128 --exclude $(PEP8EXCLUDE) . || true
	@echo "-----------"
	@echo "Ignored in PEP8 check:"
	@echo $(PEP8EXCLUDE)
