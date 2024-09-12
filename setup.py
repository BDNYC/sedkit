#!/usr/bin/env python

# allows you to build sphinx docs from the package
# main directory with "python setup.py build_sphinx"

# try:
#     from sphinx.cmd.build import build_main
#     from sphinx.setup_command import BuildDoc

#     class BuildSphinx(BuildDoc):
#         """Build Sphinx documentation after compiling C source files"""

#         description = 'Build Sphinx documentation'

#         def initialize_options(self):
#             BuildDoc.initialize_options(self)

#         def finalize_options(self):
#             BuildDoc.finalize_options(self)

#         def run(self):
#             build_cmd = self.reinitialize_command('build_ext')
#             build_cmd.inplace = 1
#             self.run_command('build_ext')
#             build_main(['-b', 'html', './docs', './docs/_build/html'])

# except ImportError:
#     class BuildSphinx(Command):
#         user_options = []

#         def initialize_options(self):
#             pass

#         def finalize_options(self):
#             pass

#         def run(self):
#             print('!\n! Sphinx is not installed!\n!', file=sys.stderr)
#             exit(1)

# DOCS_REQUIRE = [
#     'nbsphinx',
#     'sphinx',
#     'sphinx-automodapi',
#     'sphinx-rtd-theme',
#     'stsci-rtd-theme',
#     'extension-helpers',
# ]
# TESTS_REQUIRE = [
#     'pytest',
# ]