from datasets import Dataset, load_dataset , DatasetDict

def create_dataset(df, tokenizer) :

    def prepocess(samples):

        batch = []

        for instruction, input, output in zip(samples['instruction'], samples['input'], samples['output']):

            chat_prompt = [
                { "role": "user", "content": f"{instruction}\n{input}" },
                { "role": "assistant", "content": f"{output}" }
            ]

            # tokenizer 적용 후 batch에 추가
            conversation = tokenizer.apply_chat_template(chat_prompt, tokenize=False)
            conversation = conversation.rstrip() + tokenizer.eos_token
            batch.append(conversation)

        return {'text' : batch}


    def generate_dict(df) :
        instuction_list = [ open('./data/instruction.txt').read() for _ in range(len(df)) ]
        input_list = df['Question']
        output_list = df['Answer']
        dataset_dict = {'instruction' : instuction_list , 'input' : input_list , 'output' : output_list}
        dataset = Dataset.from_dict(dataset_dict)
        return dataset

    dataset = generate_dict(df)
    raw_dataset = DatasetDict()
    datasets = dataset.train_test_split(test_size=0.005 ,
                                       shuffle=True ,
                                       seed=42)

    raw_dataset['train'] = datasets['train']
    raw_dataset['test'] = datasets['test']

    raw_dataset = raw_dataset.map(prepocess,
                                  batched = True,
                                  remove_columns = raw_dataset['train'].column_names
                                  )
    train_data = raw_dataset['train']
    valid_data = raw_dataset['test']

    print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")
    print(f"A sample of train dataset: {train_data[0]}")

    return train_data, valid_data 