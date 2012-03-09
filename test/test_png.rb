require 'rubygems'
require 'minitest/autorun'
require 'oil'
require 'stringio'
require 'helper'

module Oil
  class TestPNG < MiniTest::Unit::TestCase
    # http://garethrees.org/2007/11/14/pngcrush/
    PNG_DATA = "\
\x89\x50\x4E\x47\x0D\x0A\x1A\x0A\x00\x00\x00\x0D\x49\x48\x44\x52\x00\x00\x00\
\x01\x00\x00\x00\x01\x01\x00\x00\x00\x00\x37\x6E\xF9\x24\x00\x00\x00\x10\x49\
\x44\x41\x54\x78\x9C\x62\x60\x01\x00\x00\x00\xFF\xFF\x03\x00\x00\x06\x00\x05\
\x57\xBF\xAB\xD4\x00\x00\x00\x00\x49\x45\x4E\x44\xAE\x42\x60\x82"

    def test_valid
      validate_png resize_string(PNG_DATA)
    end

    def test_bogus_header_chunk
      str = PNG_DATA.dup
      str[15] = "\x10"
      assert_raises(RuntimeError) { resize_string(str) }
    end

    def test_bogus_body_chunk
      str = PNG_DATA.dup
      str[37] = "\x10"
      assert_raises(RuntimeError) { resize_string(str) }
    end

    def test_bogus_end_chunk
      str = PNG_DATA.dup
      str[-6] = "\x10"
      if(RUBY_PLATFORM =~ /java/)
        validate_png resize_string(str) # java is fine with a bogus end chunk
      else
        assert_raises(RuntimeError) { resize_string(str) }
      end
    end

    def test_calls_each_during_yield
      io = StringIO.new(PNG_DATA)
      j = PNG.new(io, 600, 600)
      assert_raises(RuntimeError) do
        j.each{ |d| j.each { |e| j.each { |f| } } }
      end
    end
    
    def test_alloc_each
      io = PNG.allocate
      assert_raises(NoMethodError){ io.each{ |f| } }
    end

    # Test dimensions
    
    def test_zero_dim
      assert_raises(ArgumentError){ resize_string(PNG_DATA, 0, 0) }
    end
    
    def test_neg_dim
      assert_raises(ArgumentError){ resize_string(PNG_DATA, -1231, -123) }
    end
    
    # Test io source handler
    
    def test_io_returns_too_much_data
      proc = Proc.new do |io, size, buf|
        str = io.read(size)
        str = str[0..-2] * 2 if str
        if buf
          buf.slice!(0,0) # this empties the buffer
          buf << str
        else
          str
        end
      end
      assert_raises(RuntimeError) { custom_io proc, PNG_DATA }
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
        if buf
          parent.read(size, buf)
        else
          parent.read(size)
        end
      end
      assert_raises(CustomError) { custom_io proc, big_png }
    end

    def test_io_throws_in_body
      flag = false
      proc = Proc.new do |parent, size, buf|
        throw :foo if flag
        flag = true
        if buf
          parent.read(size, buf)
        else
          parent.read(size)
        end
      end
      catch(:foo){ custom_io proc, big_png }
    end

    def test_io_shrinks_buffer
      proc = Proc.new do |parent, size, buf|
        if buf
          parent.read(size, buf)
          buf.slice!(0, 10)
        else
          res = parent.read(size)
          res[10, -1]
        end
      end
      assert_raises(RuntimeError) { custom_io(proc, big_png) }
    end

    def test_io_enlarges_buffer_with_eof
      proc = Proc.new do |parent, size, buf|
        if buf
          parent.read(size, buf)
          buf << buf
        else
          str = parent.read(size)
          str * 2
        end
      end
      
      assert_raises(RuntimeError) { custom_io(proc, big_png) }
    end

    # Test yielding

    def test_raise_in_each
      assert_raises(CustomError) do
        io = StringIO.new(PNG_DATA)
        PNG.new(io, 200, 200).each { raise CustomError }
      end
    end

    def test_throw_in_each
      catch(:foo) do
        io = StringIO.new(PNG_DATA)
        PNG.new(io, 200, 200).each { throw :foo }
      end
    end

    def test_each_shrinks_buffer
      io = StringIO.new(PNG_DATA)
      io_out = binary_stringio
      PNG.new(io, 200, 200).each { |d| io_out << d; d.slice!(0, 4) }
      validate_png(io_out.string)
    end
    
    def test_each_enlarges_buffer
      io = StringIO.new(PNG_DATA)
      io_out = binary_stringio
      PNG.new(io, 200, 200).each { |d| io_out << d; d << "foobar" }
      validate_png(io_out.string)
    end

    private
    
    def io(io, width=nil, height=nil)
      width ||= 100
      height ||= 200
      out = binary_stringio
      PNG.new(io, width, height).each{ |d| out << d }
      return out.string
    end

    def custom_io(*args)
      io CustomIO.new(*args)
    end

    def resize_string(str, width=nil, height=nil)
      io(StringIO.new(str), width, height)
    end
    
    def validate_png(data)
      assert_equal "\x89\x50\x4E\x47\x0D\x0A\x1A\x0A", data[0,8]
      assert_equal "\x49\x45\x4E\x44\xAE\x42\x60\x82", data[-8, 8]
    end

    def big_png
      resize_string(PNG_DATA, 200, 200)
    end

    def binary_stringio
      io = StringIO.new
      io.set_encoding 'ASCII-8BIT' if RUBY_VERSION >= '1.9'
      io
    end
  end
end
