from datasets import Dataset, load_dataset , DatasetDict

def create_dataset(df, tokenizer , chat_template = False) : 
    
    def prepocess(samples):
        
        batch = []
        
        if chat_template : 
            
            chat_prompt = [ 
                { "role" : "system" , "content" : {instruction}} ,
                { "role" : "user" , "content" : {input} } ,
                { "role" : "system" , "content" : {output} }
            ]
        
            for instruction , input , output in zip(samples['instruction'] , samples['input'] , samples['output']) :
                prompt_template = [ item['content'].format(instruction=instruction, input=input, output=output) for item in chat_prompt]
                conversation = tokenizer.apply_chat_template(prompt_template , tokenize = False)
                batch.append(conversation)
        
            return {'text' : batch}
        
        else : 
            alpaca_prompt = (
                    "Below is an instruction that describes a task, paired with an input that provides further context.\n"
                    "아래는 작업을 설명하는 명령어와 추가적 맥락을 제공하는 입력이 짝을 이루는 예제입니다.\n\n"
                    "Write a response that appropriately completes the request.\n요청을 적절히 완료하는 응답을 작성하세요.\n\n"
                    "### Instruction(명령어):{instruction}\n\n### Input(입력):{input}\n\n### Response(응답):{response}"
            )
            
            for instruction, input, output in zip(samples["instruction"], samples["input"], samples["output"]):
                user_input = input 
                response = output + tokenizer.eos_token
                conversation = alpaca_prompt['prompt_input'].replace('{instruction}', instruction[0]).replace('{input}', user_input).replace('{response}', response) 
                batch.append(conversation)
        
            return {"content": batch}
            
    
    def generate_dict(df) :
        instuction_list = [ open('./data/instruction.txt').read() for _ in range(len(df)) ]
        input_list = df['Question']
        output_list = df['Answer']
        dataset = {'instruction' : instuction_list , 'input' : input_list , 'output' : output_list}
        return dataset
    
    dataset = generate_dict(df)
    raw_dataset = DatasetDict()
    dataset = dataset.train_test_split(test_size=0.005 , 
                                       shuffle=True , 
                                       seed=42) 
    
    raw_dataset['train'] = dataset['train']
    raw_dataset['test'] = dataset['test'] 
    
    raw_dataset = raw_dataset.map(prepocess,
                                  batched = True,
                                  remove_columns = raw_dataset['train'].column_names 
                                  )
    train_data = raw_dataset['train']
    valid_data = raw_dataset['test']
    
    print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")
    print(f"A sample of train dataset: {train_data[0]}")

    return train_data, valid_data  