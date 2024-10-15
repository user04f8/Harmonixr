# Harmonixr
*Generative MIDI Transformer for Note Infill / Music Generation*

## Executive Summary
We propose developing a Transformer-based model to assist musicians and producers by automating the note infill process in music production environments. By incorporating rhythm and dynamics into the embedding space, we aim to model MIDI more generally and therefore produce more musically coherent output, thereby facilitating a fuller realization of a creative vision that align with a user's artistic intentions.

## Project Description
Music production involves repetitive tasks such as note placement and variation, which can be time-consuming and hinder creative flow, especially in the highly spontaneous nature of producing music. This project leverages the transformer architecture for a generative model that intelligently infills notes based on existing musical context. The model will incorporate a more general embedding space, such as rhythm and dynamics, to ensure that generated notes are not only harmonically appropriate but also rhythmically and dynamically consistent with the context.

## Goals
We aim to develop a Transformer-based generative model for note infill that integrates rhythm and dynamics into the embedding space. We also hope to build a foundation extensible for future model refinements, including reinforcement learning from human feedback (RLHF) to further align the model with user preferences and grounding model outputs on other context, such as text predicates.

## Data Collection
- **Source**: Utilize the [Lakh MIDI Dataset](https://colinraffel.com/projects/lmd/) and other publicly available MIDI files covering diverse genres to ensure model generalization.
- **Features**:
  - **Pitch**: Sequences of MIDI note numbers.
  - **Velocity**: Dynamics information representing note intensity.
  - **Timing**: Tempo and timing events capturing rhythmic patterns.
  - **Duration**: Length of each note.
- **Human Feedback (Future Work)**: We also propose collecting human feedback over a longer period of time beyond the scope of this project will enable future work extending/finetuning our model via RLHF.

## Data Cleaning
- **Standardization**: Normalize all MIDI files to extract the above features using `mido` or a similar package, rejecting invalid midi files.
- **Quantization**: Align note events to fixed time steps to facilitate uniform sequence modeling.
- **Data Augmentation**: Apply transposition to different keys to increase dataset variability and improve model robustness, in addition to other augmentation and regularization to reduce overfitting.

## Feature Extraction
- **Enhanced Embeddings**: Develop embeddings/tokenization that encode pitch, velocity, and timing information, providing a comprehensive embedding space for the transformer.
- **Sequence Preparation**: Structure data into input-output pairs where the model predicts subsequent notes based on preceding sequences.

## Modelling
- **Model Architecture**: Implement a Transformer model. We will test various architectures, such as encoder/decoder layers and decoder-only networks.
- **Training Approach**:
  - **Supervised Learning**: Train the model on prepared MIDI sequences to learn the probability distribution of subsequent notes.
  - **Reinforcement Learning (Future Work)**: Incorporate RLHF to fine-tune the model based on human preferences.

## Visualization
- **Embedding Space Analysis**: Utilize t-SNE or PCA to visualize how the model's embeddings represent different musical attributes.
- **Generated Music Visualization**: Implement interactive piano rolls to compare original and generated note sequences. 

We also propose the best way of understanding model output is auditory rather than visual; we propose listening to model outputs as a means of qualitatively understanding model behavior.

## Test Plan / Metrics
- **Data Split**: We will split some percent of our data as validation data to test as we train the model; in addition, we will set aside or collect further test data to later evaluate our model without overfitting the modelling approach itself to the validation data.
- **Evaluation Metrics**:
  - **Validation Loss**: Monitor cross-entropy loss to assess predictive accuracy.
  - **Musical Coherence**: Conduct qualitative evaluations to assess the musicality of generated notes.
  - **RLHF Metrics (Future Steps)**: Track human feedback scores and reward signals to evaluate the effectiveness of reinforcement learning fine-tuning.
