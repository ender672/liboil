require 'rubygems'
require 'minitest/autorun'
require 'oil'
require 'stringio'
require 'helper'

class TestPPM < MiniTest::Unit::TestCase
  # Header Tests

  def test_invalid_dimensions
    [0, 123_314_234, -5].each do |i|
      io = PPM.new(i, 10).header_to_io
      assert_raises(RuntimeError, "#{i} is an invalid width dimension.") do
        Oil.new(io, 10, 10)
      end

      io = PPM.new(10, i).header_to_io
      assert_raises(RuntimeError, "#{i} is an invalid height dimension.") do
        Oil.new(io, 10, 10)
      end
    end
  end

  def test_onebyone
    o = Oil.new(PPM.new(1, 1).to_io, 10, 10)
    assert_equal(1, o.width)
    assert_equal(1, o.height)
  end

  def test_dimensions
    [1, 2, 3, 1000, 10000].each do |i|
      io = PPM.new(i, 10).header_to_io
      o = Oil.new(io, 10, 10)
      assert_equal(i, o.width)
      assert_equal(10, o.height)

      io = PPM.new(10, i).header_to_io
      o = Oil.new(io, 10, 10)
      assert_equal(i, o.height)
      assert_equal(10, o.width)
    end
  end

  def test_bad_signature
    ["A3 12 12 255 ", "P3 12 12 255 ", "\x006 12 12 255 ", "P612 12 255 ",
     "P6\x0012 12 255"].each do |s|
      io = StringIO.new(s)
      assert_raises(RuntimeError, "#{s} is not a valid signature") { Oil.new(io, 10, 10) }
    end
  end

  def test_unsupported_max
    [0, 1, -3, 256, 1023, 123_456_789_012].each do |i|
      io = StringIO.new("P6 12 12 #{i} ")
      assert_raises(RuntimeError) { Oil.new(io, 10, 10) }
    end
  end

  def test_header_whitespace
    io = StringIO.new("P6  22  33  255 ")
    o = Oil.new(io, 10, 10)
    assert_equal(22, o.width)
    assert_equal(33, o.height)

    io = StringIO.new("P6 \t\n 44 \t\t 55\t255 ")
    o = Oil.new(io, 10, 10)
    assert_equal(44, o.width)
    assert_equal(55, o.height)
  end

  def test_header_too_much_whitespace
    str = "P6 22 33 #{' ' * 10_000} 255 "
    io = StringIO.new(str)
    assert_raises(RuntimeError) { Oil.new(io, 10, 10) }
  end

  def test_header_truncated
    h = PPM.new(22, 33).header
    (h.size - 1).times do |i|
      s = h[0..i]
      io = StringIO.new(s)
      assert_raises(RuntimeError, "#{s} is an invalid header") { Oil.new(io, 10, 10) }
    end
  end

  # Body test

  def test_pixel_values
    ppm = PPM.new(10, 20)
    out = ''

    Oil.new(ppm.to_io, 10, 20).each{ |d| out << d }
    assert_equal(ppm.to_s, out)
  end

  # Allocation tests

  def test_multiple_initialize_leak
    o = Oil.allocate

    io = PPM.new(10, 10).to_io
    o.send(:initialize, io, 6, 7)
    o.each{ |d| }

    io = PPM.new(20, 15).to_io
    o.send(:initialize, io, 8, 9)
    o.each{ |d| }
  end

  def test_each_after_allocate
    o = Oil.allocate
    assert_raises(RuntimeError){ o.each{ |d| } }
  end

  def test_source_dimensions_after_allocate
    o = Oil.allocate
    assert_raises(RuntimeError){ o.width }
  end

  # Test io

  IO_OFFSETS = [0, 10, 20, 1023, 1024, 1025, 1026, 2000, 6000, 15000]
  PPM_STR = PPM.new(90, 100).to_s

  def iotest(io_class)
    IO_OFFSETS.each do |i|
      yield io_class.new(PPM_STR, :byte_count => i)
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
      assert_raises(RuntimeError) { resize(io) }
    end
  end

  # Test yielding

  def test_raise_in_each
    assert_raises(CustomError) do
      io = PPM.new(10, 20).to_io
      Oil.new(io, 10, 20).each { raise CustomError }
    end
  end

  def test_throw_in_each
    catch(:foo) do
      io = PPM.new(10, 20).to_io
      Oil.new(io, 10, 20).each { throw :foo }
    end
  end

  def test_each_in_each
    io = PPM.new(10, 10).to_io
    o = Oil.new(io, 10, 20)
    o.each do |d|
      assert_raises(RuntimeError){ o.each { |e| } }
    end
  end
end
