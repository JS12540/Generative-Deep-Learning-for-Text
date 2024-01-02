# Generative Deep Learning for Text

## Overview

This project focuses on building a language model using a generative deep learning approach. The model is implemented using a Transformer architecture with TensorFlow/Keras, aiming to generate creative and coherent text based on the learned patterns in the training data.

## Code Structure

The project code consists of several components:

1. **Text Preprocessing:**
   - Utilizes the `TextVectorization` layer to convert text data into sequences of integers.
   - Implements a custom dataset preparation function (`prepare_lm_dataset`) to create inputs and targets for language modeling.

2. **Positional Embedding Layer:**
   - Introduces a `PositionalEmbedding` layer to add positional information to input sequences.

3. **Transformer Decoder:**
   - Defines a `TransformerDecoder` layer representing a single transformer block with multi-head self-attention and cross-attention mechanisms.

4. **Model Construction:**
   - Constructs a language model using the previously defined layers.
   - Compiles the model using the sparse categorical crossentropy loss and the RMSprop optimizer.

5. **Text Generation:**
   - Implements a `TextGenerator` callback that samples text from the trained model at the end of each epoch.
   - Allows for variable temperature sampling to control the randomness of the generated text.

## Training

To train the model, use the provided language modeling dataset (`lm_dataset`). The training is set to run for 200 epochs, and the `TextGenerator` callback is included to observe the generated text at the end of each epoch.

```python
model.fit(lm_dataset, epochs=200, callbacks=[text_gen_callback])
```

## Text Generation Observations

The generated text showcases the impact of temperature on the diversity and creativity of the language model. As mentioned in the code, a low temperature results in repetitive text, while higher temperatures lead to more interesting and creative outputs. A recommended generation temperature of about 0.7 provides a balanced mix of learned structure and randomness.

> As you can see, a low temperature value results in very boring and repetitive text and can sometimes cause the generation process to get stuck in a loop. With higher temperatures, the generated text becomes more interesting, surprising, even creative. With a very high temperature, the local structure starts to break down, and the output looks largely random. Here, a good generation temperature would seem to be about 0.7. Always experiment with multiple sampling strategies! A clever balance between learned structure and randomness is what makes generation interesting.

## GPT-3 Comparison

It's worth noting that GPT-3, a state-of-the-art language model, shares similarities with the architecture trained in this example. GPT-3 employs a deep stack of Transformer decoders and a significantly larger training corpus.

Feel free to explore and experiment with the provided code to enhance your understanding of generative deep learning for text!
