class CustomError < RuntimeError; end

class CustomIO
  def initialize(proc, *args)
    @parent = StringIO.new(*args)
    @proc = proc
  end

  def read(size, buf=nil)
    @proc.call(@parent, size, buf) if @proc
  end
end
