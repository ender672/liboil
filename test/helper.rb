require 'stringio'

class CustomError < RuntimeError; end

class CustomIO < StringIO
  def initialize(str, options={})
    @read_count = options[:read_count]
    @byte_count = options[:byte_count]
    @read_count = 0 unless @read_count || @byte_count
    super(str)
  end

  def countdown(size)
    ret = (@read_count && @read_count <= 0) || (@byte_count && @byte_count <= 0)
    @read_count -= 1 if @read_count
    @byte_count -= size if @byte_count
    ret
  end
end

class GrowIO < CustomIO
  def read(*args)
    size = args[0]
    buf = args[1]
    return super unless countdown(size)
    str = super(size)
    str = str[0..-2] * 3 if str

    if buf
      buf.slice!(0,0) # this empties the buffer
      buf << str
    else
      str
    end
  end
end

class ShrinkIO < CustomIO
  def read(*args)
    size = args[0]
    return super unless countdown(size)
    res = super(size)
    raise RuntimeError unless res
    new_size = res.size / 2
    res[-1 * new_size, new_size]
  end
end

class RaiseIO < CustomIO
  def read(*args)
    return super unless countdown(size)
    raise CustomError
  end
end

class ThrowIO < CustomIO
  def read(*args)
    return super unless countdown(size)
    throw(:foo)
  end
end

class NilIO < CustomIO
  def read(*args)
    super unless countdown(size)
  end
end

class NotStringIO < CustomIO
  def read(*args)
    return super unless countdown(size)
    return 78887
  end
end

class PPM
  def initialize(w, h)
    @w = w
    @h = h
  end

  def rowlen
    @w * 3
  end

  def row_samples(row)
    (0...rowlen).map{ |i| (((row - 1) * @w + i) % 255).chr }.join
  end

  def samples
    (1..@h).map{ |i| row_samples(i) }.join
  end

  def header
    "P6 #{@w} #{@h} 255 "
  end

  def to_s
    header + samples
  end

  def to_io
    StringIO.new(to_s)
  end

  def header_to_io
    StringIO.new(header)
  end

  def print
    i = 0
    samples.bytes.each_slice(rowlen) do |sl|
      Kernel.print "#{i}:"
      sl.each{ |s| Kernel.print '%4d' % s }
      puts ""
      i += 1
    end
  end
end

def resize_string(str, width=nil, height=nil)
  io = StringIO.new(str)
  width ||= 100
  height ||= 200
  out = binary_stringio
  Oil.new(io, width, height).each{ |d| out << d }
  out.string
end

def binary_stringio
  io = StringIO.new
  io.set_encoding 'ASCII-8BIT' if RUBY_VERSION >= '1.9'
  io
end

