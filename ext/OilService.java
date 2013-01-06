import java.awt.Image;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.Iterator;
import javax.imageio.ImageIO;
import javax.imageio.ImageReader;
import javax.imageio.stream.ImageInputStream;
import org.jruby.*;
import org.jruby.anno.JRubyMethod;
import org.jruby.runtime.Block;
import org.jruby.runtime.ObjectAllocator;
import org.jruby.runtime.ThreadContext;
import org.jruby.runtime.builtin.IRubyObject;
import org.jruby.runtime.load.BasicLibraryService;
import org.jruby.util.IOInputStream;

public class OilService implements BasicLibraryService {
    public static class OilImage extends RubyObject {
        private int in_width, in_height, out_width, out_height;
        private ImageReader reader;
        private boolean each_called;

        public OilImage(Ruby runtime, RubyClass klass) {
            super(runtime, klass);
            reader = null;
            in_width = in_height = out_width = out_height = 0;
            each_called = false;
        }

        @JRubyMethod
        public RubyFixnum width(ThreadContext context) {
            Ruby runtime = context.runtime;
            return runtime.newFixnum(in_width);
        }

        @JRubyMethod
        public RubyFixnum height(ThreadContext context) {
            Ruby runtime = context.runtime;
            return runtime.newFixnum(in_height);
        }

        @JRubyMethod
        public IRubyObject initialize(IRubyObject io, IRubyObject rb_width, IRubyObject rb_height) {
            ImageInputStream iis;
            Iterator readers;

            each_called = false;
            out_width = RubyFixnum.num2int(rb_width);
            out_height = RubyFixnum.num2int(rb_height);

            if (out_width < 1 || out_height < 1)
                throw getRuntime().newArgumentError("dimensions must be > 0");

            try {
                iis = ImageIO.createImageInputStream(new IOInputStream(io));
                readers = ImageIO.getImageReaders(iis);
                if (!readers.hasNext()) {
                    throw getRuntime().newRuntimeError("Image type not recognized.");
                }
                reader = (ImageReader)readers.next();
                reader.setInput(iis, true);

                in_width = reader.getWidth(0);
                in_height = reader.getHeight(0);
            }
            catch(IOException ioe) {
                throw getRuntime().newRuntimeError("error");
            }
            catch(ArrayIndexOutOfBoundsException iob) {
                throw getRuntime().newRuntimeError("error");
            }

            return this;
        }

        @JRubyMethod
        public IRubyObject each(ThreadContext context, Block block) {
            Image newImg;

            if (reader == null)
                throw getRuntime().newNoMethodError("each Called before initializing", null, context.getRuntime().getNil());

            if (each_called)
                throw getRuntime().newRuntimeError("each called twice.");
            each_called = true;

            try {
                double x = (double)out_width / in_width;
                double y = (double)out_height / in_height;
                if (x < y) out_height = (int)(in_height * x);
                else out_width = (int)(in_width * y);
                if (out_height < 1) out_height = 1;
                if (out_width < 1) out_width = 1;

                newImg = reader.read(0).getScaledInstance(out_width, out_height, Image.SCALE_SMOOTH);
                BufferedImage bim = new BufferedImage(out_width, out_height, BufferedImage.TYPE_INT_RGB);
                bim.createGraphics().drawImage(newImg, 0, 0, null);

                ByteArrayOutputStream baos = new ByteArrayOutputStream();
                ImageIO.write(bim, reader.getFormatName(), baos);

                block.yield(context, new RubyString(getRuntime(), getRuntime().getString(), baos.toByteArray()));
            }
            catch(IOException ioe) {
                throw getRuntime().newRuntimeError("error");
            }
            catch(ArrayIndexOutOfBoundsException iob) {
                throw getRuntime().newRuntimeError("error");
            }

            return this;
        }
    }

    private static ObjectAllocator OIL_ALLOCATOR = new ObjectAllocator() {
        public IRubyObject allocate(Ruby runtime, RubyClass klass) {
            return new OilImage(runtime, klass);
        }
    };

    public boolean basicLoad(Ruby runtime) {
        RubyClass oil = runtime.defineClass("Oil", runtime.getObject(), OIL_ALLOCATOR);
        oil.setConstant("JPEG", oil);
        oil.setConstant("PNG", oil);
        oil.defineAnnotatedMethods(OilImage.class);
        return true;
    }
}
