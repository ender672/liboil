liboil
======

A C library for resizing JPEG, PNG, and PPM images. It aims for fast performance
and low memory use.

Requirements
------------

libjpeg development headers and libpng development headers.

Installation
------------

Running make will generate a static library, liboil.a:

$ make

Usage
-----

Make sure to search the system jpeg & png libraries when linking:

cc myobject.o path/to/liboil.a -ljpeg -lpng

Testing
-------

You can compile the sample binary "resize" to test and use this library:

$ make resize
$ ./resize --in infile.jpg --out outfile.jpg --width 200 --height 200

There are other tests that you can run from the makefile. This test will
download a 10,000x10,000 pixel jpeg and resize it:

$ make test

License
--------

(The MIT License)

Copyright (c) 2014

* {Timothy Elliott}[http://holymonkey.com]

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
'Software'), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
