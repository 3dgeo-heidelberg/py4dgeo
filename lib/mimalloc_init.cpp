#include <mimalloc.h>
#pragma comment(                                                               \
    linker, "/include:mi_version") // ensures linker includes mimalloc.lib

void
force_mimalloc()
{
  mi_version(); // force DLL to load and init
}
