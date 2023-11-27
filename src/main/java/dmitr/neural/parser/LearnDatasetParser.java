package dmitr.neural.parser;

import dmitr.neural.LearnDataset;

import java.io.*;
import java.nio.charset.StandardCharsets;

public class LearnDatasetParser implements IParser<LearnDataset> {

    @Override
    public void parseOut(LearnDataset dataset, OutputStream stream) {
        try (BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(stream, StandardCharsets.UTF_8))) {
            writer.write(dataset.getInputNeurons() + "\n");
            writer.write(dataset.getOutputNeurons() + "\n");

            for (double[] kit : dataset.getSet())
                for (double value : kit)
                    writer.write(value + "\n");
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public LearnDataset parseIn(InputStream stream) {
        LearnDataset dataset = null;

        try (BufferedReader reader = new BufferedReader(new InputStreamReader(stream, StandardCharsets.UTF_8))) {
            int inputNeurons = Integer.parseInt(reader.readLine());
            int outputNeurons = Integer.parseInt(reader.readLine());

            dataset = new LearnDataset(inputNeurons, outputNeurons);

            double[] set = new double[inputNeurons + outputNeurons];
            int factor = 0;

            for (String line = reader.readLine(); line != null; line = reader.readLine()) {
                set[factor] = Double.parseDouble(line);
                System.out.println(line);
                factor++;
                if (factor == set.length) {
                    factor = 0;
                    dataset.insert(set.clone());
                }
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        return dataset;
    }

}
