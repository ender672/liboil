require 'rubygems'
require 'minitest/autorun'
require 'oil'
require 'stringio'
require 'helper'

class TestJPEG < MiniTest::Unit::TestCase
  # http://stackoverflow.com/a/2349470
  JPEG_DATA = "\
\xff\xd8\xff\xe0\x00\x10\x4a\x46\x49\x46\x00\x01\x01\x01\x00\x48\x00\x48\x00\
\x00\xff\xdb\x00\x43\x00\x03\x02\x02\x02\x02\x02\x03\x02\x02\x02\x03\x03\x03\
\x03\x04\x06\x04\x04\x04\x04\x04\x08\x06\x06\x05\x06\x09\x08\x0a\x0a\x09\x08\
\x09\x09\x0a\x0c\x0f\x0c\x0a\x0b\x0e\x0b\x09\x09\x0d\x11\x0d\x0e\x0f\x10\x10\
\x11\x10\x0a\x0c\x12\x13\x12\x10\x13\x0f\x10\x10\x10\xff\xc9\x00\x0b\x08\x00\
\x01\x00\x01\x01\x01\x11\x00\xff\xcc\x00\x06\x00\x10\x10\x05\xff\xda\x00\x08\
\x01\x01\x00\x00\x3f\x00\xd2\xcf\x20\xff\xd9"

  BIG_JPEG = begin
    resize_string(JPEG_DATA, 2000, 2000)
  end

  def test_valid
    o = Oil.new(jpeg_io, 10 ,20)
    assert_equal 1, o.width
    assert_equal 1, o.height
  end

  def test_missing_eof
    io = StringIO.new(JPEG_DATA[0..-2])
    o = Oil.new(io, 10 ,20)
    assert_equal 1, o.width
    assert_equal 1, o.height
  end

  def test_bogus_header_marker
    str = JPEG_DATA.dup
    str[3] = "\x10"
    assert_raises(RuntimeError) { resize_string(str) }
  end

  def test_bogus_body_marker
    str = JPEG_DATA.dup
    str[-22] = "\x10"
    assert_raises(RuntimeError) { resize_string(str) }
  end

  # Allocation tests

  def test_multiple_initialize_leak
    o = Oil.allocate

    o.send(:initialize, jpeg_io, 6, 7)
    o.each{ |d| }

    o.send(:initialize, jpeg_io, 8, 9)
    o.each{ |d| }
  end

  # Test io

  IO_OFFSETS = [0, 10, 20, 8191, 8192, 8193, 12000]

  def iotest(io_class)
    IO_OFFSETS.each do |i|
      yield io_class.new(BIG_JPEG, :byte_count => i)
    end
  end

  def resize(io)
    o = Oil.new(io, 20, 10)
    o.each{ |d| }
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
      resize(io) rescue nil
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
      Oil.new(jpeg_io, 10, 20).each { raise CustomError }
    end
  end

  def test_throw_in_each
    catch(:foo) do
      Oil.new(jpeg_io, 10, 20).each { throw :foo }
    end
  end

  def test_each_in_each
    o = Oil.new(jpeg_io, 10, 20)
    o.each do |d|
      assert_raises(RuntimeError){ o.each { |e| } }
    end
  end

  def test_each_shrinks_buffer
    io = StringIO.new(JPEG_DATA)
    io_out = binary_stringio
    Oil.new(io, 200, 200).each { |d| io_out << d; d.slice!(0, 4) }
  end
  
  def test_each_enlarges_buffer
    io = StringIO.new(JPEG_DATA)
    io_out = binary_stringio
    Oil.new(io, 200, 200).each { |d| io_out << d; d << "foobar" }
  end

  private

  def jpeg_io
    StringIO.new(JPEG_DATA)
  end
end
