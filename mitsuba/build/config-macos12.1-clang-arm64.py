BUILDDIR       = '#build/release'
DISTDIR        = '#Mitsuba.app'
CXX            = 'clang++'
CC             = 'clang'
CCFLAGS        = ['-mmacosx-version-min=12.1', '-funsafe-math-optimizations', '-fno-math-errno', '-isysroot', '/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX12.1.sdk', '-O3', '-Wall', '-Wno-deprecated-declarations', '-g', '-DMTS_DEBUG', '-DSINGLE_PRECISION', '-DSPECTRUM_SAMPLES=3', '-fvisibility=hidden', '-ftemplate-depth=512', '-stdlib=libc++', '-Wno-asm-operand-widths', '-Wno-deprecated-register']
LINKFLAGS      = ['-framework', 'OpenGL', '-framework', 'Cocoa', '-mmacosx-version-min=12.1', '-Wl,-syslibroot,/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX12.1.sdk', '-Wl,-headerpad,128', '-stdlib=libc++']
CXXFLAGS       = ['-std=c++11']
BASEINCLUDE    = ['#include', '/opt/homebrew/include']
BASELIBDIR     = ['/opt/homebrew/lib']
BASELIB        = ['m', 'pthread', 'Imath']
EIGENINCLUDE   = ['/opt/homebrew/include/eigen3']
OEXRINCLUDE    = ['/opt/homebrew/include/OpenEXR', '/opt/homebrew/include/Imath']
OEXRLIB        = ['Imath', 'Iex', 'z', 'OpenEXR']
PNGLIB         = ['png16']
PNGINCLUDE     = ['/opt/homebrew/include/libpng']
JPEGLIB        = ['jpeg']
JPEGINCLUDE    = ['/opt/homebrew/include/libjpeg']
XERCESLIB      = ['xerces-c']
GLLIB          = ['glew', 'objc']
GLFLAGS        = ['-DGLEW_MX']
BOOSTINCLUDE   = ['#dependencies']
BOOSTLIB       = ['boost_filesystem', 'boost_system', 'boost_thread-mt']
PYTHON27INCLUDE= ['/System/Library/Frameworks/Python.framework/Versions/2.7/Headers']
PYTHON27LIBDIR = ['/System/Library/Frameworks/Python.framework/Versions/2.7/lib']
PYTHON27LIB    = ['boost_python27', 'boost_system']
PYTHON35INCLUDE= ['#dependencies/include/python3.5']
PYTHON35LIB    = ['boost_python36', 'boost_system']
PYTHON36INCLUDE= ['#dependencies/include/python3.4']
PYTHON36LIB    = ['boost_python36', 'boost_system']
# COLLADAINCLUDE = ['#dependencies/include/collada-dom', '#dependencies/include/collada-dom/1.4']
# COLLADALIB     = ['collada14dom24']
QTDIR          = '/opt/homebrew/Cellar/qt@5/5.15.2_1'
FFTWLIB        = []#'fftw3']