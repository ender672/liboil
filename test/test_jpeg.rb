require 'rubygems'
require 'minitest/autorun'
require 'oil'
require 'stringio'

module Oil
  class TestJPEG < MiniTest::Unit::TestCase
    # http://www.techsupportteam.org/forum/digital-imaging-photography/1892-worlds-smallest-valid-jpeg.html
    JPEG_DATA = "\
\xFF\xD8\xFF\xE0\x00\x10\x4A\x46\x49\x46\x00\x01\x01\x01\x00\x48\x00\x48\x00\
\x00\xFF\xDB\x00\x43\x00\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\
\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\
\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\
\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xC2\x00\x0B\x08\x00\
\x01\x00\x01\x01\x01\x11\x00\xFF\xC4\x00\x14\x10\x01\x00\x00\x00\x00\x00\x00\
\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xFF\xDA\x00\x08\x01\x01\x00\x01\x3F\
\x10\xFF\xD9"

    def test_valid_jpeg
      validate_jpeg resize_string(JPEG_DATA)
    end

    def test_jpeg_missing_eof
      validate_jpeg resize_string(JPEG_DATA[0..-2])
    end

    def test_jpeg_bogus_header_marker
      str = JPEG_DATA.dup
      str[3] = "\x10"
      assert_raises(RuntimeError) { resize_string(str) }
    end

    def test_jpeg_bogus_body_marker
      str = JPEG_DATA.dup
      str[-1] = "\x10"
      assert_raises(RuntimeError) { resize_string(str) }
    end

    def test_calls_each_during_yield
      io = StringIO.new(JPEG_DATA)
      j = JPEG.new(io, 600, 600)
      assert_raises(RuntimeError) do
        j.each{ |d| j.each { |e| j.each { |f| } } }
      end
    end
    
    # Test io source handler
    
    def test_io_returns_too_much_data
      proc = Proc.new do |io, size, buf|
        buf.slice!(0,0)
        buf << (io.read(size)[0..-2] * 2)
      end
      assert_raises(RuntimeError) { custom_io proc, JPEG_DATA }
      #custom_io proc, JPEG_DATA
    end
    
    def test_io_does_nothing
      assert_raises(RuntimeError) { custom_io(nil) }
    end

    def test_io_raises_exception_immediately
      proc = Proc.new{ raise CustomError }
      assert_raises(CustomError) { custom_io proc }
    end

    def test_io_throws_immediately
      proc = Proc.new{ throw(:foo) }
      catch(:foo){ custom_io proc }
    end
    
    def test_io_raises_exception_in_body
      flag = false
      proc = Proc.new do |parent, size, buf|
        raise CustomError if flag
        flag = true
        parent.read(size, buf)
      end
      assert_raises(CustomError) { custom_io proc, big_jpeg }
    end

    def test_io_throws_in_body
      flag = false
      proc = Proc.new do |parent, size, buf|
        throw :foo if flag
        flag = true
        parent.read(size, buf)
      end
      catch(:foo){ custom_io proc, big_jpeg }
    end

    def test_io_shrinks_buffer
      proc = Proc.new do |parent, size, buf|
        parent.read(size, buf)
        buf.slice!(0, 10)
      end
      assert_raises(RuntimeError) { custom_io(proc, big_jpeg) }
    end

    def test_io_enlarges_buffer
      proc = Proc.new do |parent, size, buf|
        res = parent.read(size, buf)
        buf << res
      end
      validate_jpeg custom_io(proc, big_jpeg) # how is this okay?
    end

    def test_io_seek
      # Craft a JPEG with header content that will be skipped.
      # Make sure an actual seek happens, and it doesn't just skip buffer.
    end

    # Test yielding

    def test_raise_in_each
      assert_raises(CustomError) do
        io = StringIO.new(JPEG_DATA)
        JPEG.new(io, 200, 200).each { raise CustomError }
      end
    end

    def test_throw_in_each
      catch(:foo) do
        io = StringIO.new(JPEG_DATA)
        JPEG.new(io, 200, 200).each { throw :foo }
      end
    end

    def test_each_shrinks_buffer
      io = StringIO.new(JPEG_DATA)
      io_out = binary_stringio
      JPEG.new(io, 200, 200).each { |d| io_out << d; d.slice!(0, 4) }
      validate_jpeg(io_out.string)
    end
    
    def test_each_enlarges_buffer
      io = StringIO.new(JPEG_DATA)
      io_out = binary_stringio
      JPEG.new(io, 200, 200).each { |d| io_out << d; d << "foobar" }
      validate_jpeg(io_out.string)
    end

    private
    
    def io(io, width=nil, height=nil)
      width ||= 100
      height ||= 200
      out = binary_stringio
      JPEG.new(io, width, height).each{ |d| out << d }
      return out.string
    end

    def custom_io(*args)
      io CustomIO.new(*args)
    end

    def resize_string(str, width=nil, height=nil)
      io(StringIO.new(str), width, height)
    end
    
    def validate_jpeg(data)
      assert_equal "\xFF\xD8", data[0,2]
      assert_equal "\xFF\xD9", data[-2, 2]
    end

    def big_jpeg
      resize_string(JPEG_DATA, 1000, 1000)
    end

    def binary_stringio
      io = StringIO.new
      io.set_encoding 'ASCII-8BIT' if RUBY_VERSION >= '1.9'
      io
    end
  end

  class CustomError < RuntimeError; end

  class CustomIO
    def initialize(proc, *args)
      @parent = StringIO.new(*args)
      @proc = proc
    end

    def read(size, buf)
      @proc.call(@parent, size, buf) if @proc
    end
  end
end

