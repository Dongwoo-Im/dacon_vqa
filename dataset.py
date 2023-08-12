import os
from PIL import Image
from torch.utils.data import Dataset


class VQADataset(Dataset):
    def __init__(self, df, processor, mode='train'):
        self.df = df
        self.mode = mode

        self.vis_processor = processor.image_processor
        self.txt_processor = processor.tokenizer

        if mode == 'test':
            self.img_path = 'data/image/test'
        else:
            self.img_path = 'data/image/train'

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_name = os.path.join(self.img_path, row['image_id'] + '.jpg')
        image = Image.open(img_name).convert('RGB')

        inputs = self.txt_processor(
            text=row['question'],
            max_length=32,
            padding='max_length',
            truncation=True,
            return_token_type_ids=False,
            return_tensors="pt",
        )
        inputs.update(
            self.vis_processor(
                images=image,
                return_tensors="pt",
            )
        )

        if self.mode != 'test':
            inputs["labels"] = self.txt_processor(
                text=row['answer'],
                max_length=32,
                padding='max_length',
                return_tensors="pt",
            ).input_ids

        for k in inputs.keys():
            inputs[k] = inputs[k].squeeze()
        
        return inputs
