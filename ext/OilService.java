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
        private IRubyObject io;
        private int width, height;

        public OilImage(Ruby runtime, RubyClass klass) {
            super(runtime, klass);
            io = runtime.getNil();
        }

        @JRubyMethod
        public IRubyObject initialize(IRubyObject rb_io, IRubyObject rb_width, IRubyObject rb_height) {
            io = rb_io;
            width = RubyFixnum.num2int(rb_width);
            height = RubyFixnum.num2int(rb_height);

            if (width < 1 || height < 1)
                throw getRuntime().newArgumentError("dimensions must be > 0");

            return this;
        }

        @JRubyMethod
        public IRubyObject each(ThreadContext context, Block block) {
            Iterator readers;
            Image newImg;
            ImageReader reader;
            ImageInputStream iis;

            if (io.isNil())
                throw getRuntime().newNoMethodError("each Called before initializing", null, context.getRuntime().getNil());

            try {
                iis = ImageIO.createImageInputStream(new IOInputStream(io));
                readers = ImageIO.getImageReaders(iis);
                if (!readers.hasNext()) {
                    throw getRuntime().newRuntimeError("Image type not recognized.");
                }
                reader = (ImageReader)readers.next();
                reader.setInput(iis, true);

                double x = (double)width / reader.getWidth(0);
                double y = (double)height / reader.getHeight(0);
                if (x < y) height = (int)(reader.getHeight(0) * x);
                else width = (int)(reader.getWidth(0) * y);
                if (height < 1) height = 1;
                if (width < 1) width = 1;

                newImg = reader.read(0).getScaledInstance(width, height, Image.SCALE_SMOOTH);
                BufferedImage bim = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
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
