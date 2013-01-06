require 'rubygems'
require 'minitest/autorun'
require 'oil'
require 'stringio'
require 'helper'

class TestPNG < MiniTest::Unit::TestCase
  # http://garethrees.org/2007/11/14/pngcrush/
  PNG_DATA = "\
\x89\x50\x4E\x47\x0D\x0A\x1A\x0A\x00\x00\x00\x0D\x49\x48\x44\x52\x00\x00\x00\
\x01\x00\x00\x00\x01\x01\x00\x00\x00\x00\x37\x6E\xF9\x24\x00\x00\x00\x10\x49\
\x44\x41\x54\x78\x9C\x62\x60\x01\x00\x00\x00\xFF\xFF\x03\x00\x00\x06\x00\x05\
\x57\xBF\xAB\xD4\x00\x00\x00\x00\x49\x45\x4E\x44\xAE\x42\x60\x82"

  BIG_PNG = begin
    resize_string(PNG_DATA, 500, 1000)
  end

  def test_valid
    o = Oil.new(png_io, 10 ,20)
    assert_equal 1, o.width
    assert_equal 1, o.height
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
    io = StringIO.new(str)
    o = Oil.new(png_io, 10 ,20)
    assert_equal 1, o.width
    assert_equal 1, o.height
  end

  # Allocation tests

  def test_multiple_initialize_leak
    o = Oil.allocate

    o.send(:initialize, png_io, 6, 7)
    o.each{ |d| }

    o.send(:initialize, png_io, 8, 9)
    o.each{ |d| }
  end

  # Test io

  IO_OFFSETS = [0, 10, 20]#, 8191, 8192, 8193, 12000]

  def iotest(io_class)
    IO_OFFSETS.each do |i|
      yield io_class.new(BIG_PNG, :byte_count => i)
    end
  end

  def resize(io)
    Oil.new(io, 20, 10).each{ |d| }
  end

  def test_io_too_much_data
    iotest(GrowIO) do |io|
      assert_raises(RuntimeError) { resize(io) }
    end
  end

  def test_io_does_nothing
    iotest(NilIO) do |io|
      assert_raises(RuntimeError) { resize(io) }
    end
  end

  def test_io_raises_exception
    iotest(RaiseIO) do |io|
      assert_raises(CustomError) { resize(io) }
    end
  end

  def test_io_throws
    iotest(ThrowIO) do |io|
      assert_throws(:foo) { resize(io) }
    end
  end

  def test_io_shrinks_buffer
    iotest(ShrinkIO) do |io|
      assert_raises(RuntimeError) { resize(io) }
    end
  end

  def test_not_string_io
    iotest(NotStringIO) do |io|
      assert_raises(TypeError) { resize(io) }
    end
  end

  # Test yielding

  def test_raise_in_each
    assert_raises(CustomError) do
      Oil.new(png_io, 10, 20).each { raise CustomError }
    end
  end

  def test_throw_in_each
    catch(:foo) do
      Oil.new(png_io, 10, 20).each { throw :foo }
    end
  end

  def test_each_in_each
    o = Oil.new(png_io, 10, 20)
    o.each do |d|
      assert_raises(RuntimeError){ o.each { |e| } }
    end
  end

  def test_each_shrinks_buffer
    io = StringIO.new(PNG_DATA)
    io_out = binary_stringio
    Oil.new(io, 200, 200).each { |d| io_out << d; d.slice!(0, 4) }
  end
  
  def test_each_enlarges_buffer
    io = StringIO.new(PNG_DATA)
    io_out = binary_stringio
    Oil.new(io, 200, 200).each { |d| io_out << d; d << "foobar" }
  end

  private

  def png_io
    StringIO.new(PNG_DATA)
  end
end
