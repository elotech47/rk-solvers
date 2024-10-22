from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools
import os
import pybind11

__version__ = '0.0.1'

class get_pybind_include(object):
    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)

ext_modules = [
    Extension(
        'rk_solver_cpp',
        [
            os.path.join('src', 'rk_solver_binding.cpp'),
            os.path.join('src', 'rk_solver.cpp'),
            os.path.join('src', 'rk23.cpp'),
            os.path.join('src', 'rk45.cpp'),
            os.path.join('src', 'rk_step.cpp'),
            os.path.join('src', 'dense_output.cpp'),
            os.path.join('src', 'solve_ivp.cpp')
        ],
        include_dirs=[
            get_pybind_include(),
            get_pybind_include(user=True),
            'include'
        ],
        language='c++'
    ),
]
 
def has_flag(compiler, flagname):
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True

def cpp_flag(compiler):
    flags = ['-std=c++17', '-std=c++14', '-std=c++11']
    for flag in flags:
        if has_flag(compiler, flag): return flag
    raise RuntimeError('Unsupported compiler -- at least C++11 support is needed!')

class BuildExt(build_ext):
    c_opts = {
        'msvc': ['/EHsc'],
        'unix': [],
    }
    l_opts = {
        'msvc': [],
        'unix': [],
    }

    if sys.platform == 'darwin':
        darwin_opts = ['-stdlib=libc++', '-mmacosx-version-min=10.7']
        c_opts['unix'] += darwin_opts
        l_opts['unix'] += darwin_opts

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        link_opts = self.l_opts.get(ct, [])
        if ct == 'unix':
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
        elif ct == 'msvc':
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
        for ext in self.extensions:
            ext.extra_compile_args = opts
            ext.extra_link_args = link_opts
        build_ext.build_extensions(self)

setup(
    name='rk_solver_cpp',
    version=__version__,
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/your_github_username/rk_solver_cpp',
    description='A C++ implementation of Runge-Kutta solvers with Python bindings',
    long_description='',
    ext_modules=ext_modules,
    install_requires=['pybind11>=2.4'],
    setup_requires=['pybind11>=2.4'],
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
)