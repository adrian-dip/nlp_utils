def add_new_tokens(model, tokenizer, new_tokens):
  
  assert type(new_tokens) == list, "New tokens must be a list"

  # delete repeated tokens
  new_tokens = set(new_tokens) - set(tokenizer.vocab.keys())

  # add the tokens 
  tokenizer.add_tokens(list(new_tokens))

  # initialize random embeddings for the new tokens
  model.resize_token_embeddings(len(tokenizer))
