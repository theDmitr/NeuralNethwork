package dmitr.neural.parser;

import java.io.InputStream;
import java.io.OutputStream;

public interface IParser<T> {

    void parseOut(T t, OutputStream stream);
    T parseIn(InputStream stream);

}
