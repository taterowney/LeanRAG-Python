from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

def fine_tune_qwen_embedding(data_generator,
                             model_name="Qwen/Qwen3-Embedding-0.6B",
                             output_path="./tuned_retriever",
                             epochs=1,
                             batch_size=32,
                             warmup_steps=100):
  """
  Fine-tunes the Qwen/Qwen3-Embedding-0.6B model on a given dataset of
  ('query', 'response') pairs.

  Args:
      data_generator (generator): A generator that yields tuples of
                                  ('query', 'response') strings.
      model_name (str, optional): The name of the SentenceTransformer model to
                                  fine-tune. Defaults to
                                  "Qwen/Qwen3-Embedding-0.6B".
      output_path (str, optional): The path to save the fine-tuned model.
                                   Defaults to "./tuned_retriever".
      epochs (int, optional): The number of epochs to train for. Defaults to 1.
      batch_size (int, optional): The batch size for training. Defaults to 32.
      warmup_steps (int, optional): The number of warmup steps for the
                                    learning rate scheduler. Defaults to 100.
  """
  # Load the pre-trained SentenceTransformer model
  model = SentenceTransformer(model_name)

  # Create a list of InputExample objects from the data generator
  train_examples = [InputExample(texts=[query, response])
                    for query, response in data_generator]

  # Create a DataLoader for the training examples
  train_dataloader = DataLoader(train_examples, shuffle=True,
                                batch_size=batch_size)

  # Define the loss function. MultipleNegativesRankingLoss is suitable for
  # ('query', 'response') pairs, as it treats other responses in the batch
  # as negative examples for a given query.
  train_loss = losses.MultipleNegativesRankingLoss(model)

  # Fine-tune the model
  model.fit(train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=warmup_steps,
            output_path=output_path)

if __name__ == '__main__':
  # Example usage with a dummy generator
  def dummy_data_generator():
    """A dummy generator for demonstration purposes."""
    data = [
        ("What is the capital of France?", "Paris is the capital of France."),
        ("Who wrote 'Hamlet'?", "William Shakespeare wrote 'Hamlet'."),
        ("What is the boiling point of water?",
         "The boiling point of water is 100 degrees Celsius."),
        ("What is the formula for water?", "The chemical formula for water is H2O."),
        ("Who painted the Mona Lisa?", "Leonardo da Vinci painted the Mona Lisa."),
    ]
    for item in data:
      yield item

  print("Starting the fine-tuning process with dummy data...")
  fine_tune_qwen_embedding(dummy_data_generator(),
                             output_path="./fine_tuned_dummy_model",
                             epochs=1,
                             batch_size=16)
  print("Fine-tuning complete. Model saved to './fine_tuned_dummy_model'.")

  # To load the fine-tuned model later:
  # fine_tuned_model = SentenceTransformer("./fine_tuned_dummy_model")