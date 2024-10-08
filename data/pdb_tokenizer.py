from transformers import PreTrainedTokenizer
import json
import re

class PDBTokenizer(PreTrainedTokenizer):
    def __init__(self, **kwargs):
        vocab = {f'{str(i)}': i for i in range(65536)}
        vocab['[CLS]'] = 257
        vocab['[BOS]'] = 258
        vocab['[EOS]'] = 259
        vocab['[UNK]'] = 260
        vocab['[SEP]'] = 261
        vocab['[UNK]'] = 262
        vocab['[PAD]'] = 263
        vocab['[MASK]'] = 264
        vocab['[Inpaint]'] = 265
        vocab['[MLM]'] = 266
        vocab['[GPT]'] = 267
        self.vocab = vocab
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
        super().__init__(**kwargs)
        self.add_special_tokens({
            'cls_token': '[CLS]',
            'bos_token': '[BOS]',
            'eos_token': '[EOS]',
            'unk_token': '[UNK]',
            'sep_token': '[SEP]',
            'pad_token': '[PAD]',
            'mask_token': '[MASK]'
        })

    @property
    def vocab_size(self):
        return len(self.vocab)

    def get_vocab(self):
        return self.vocab

    def _tokenize(self, text):
        # 使用正则表达式匹配 [token] 格式的 token
        tokens = re.findall(r'\[.*?\]|\S+', text)
        return tokens


    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.vocab['[UNK]'])


    def _convert_id_to_token(self, index):
        return self.ids_to_tokens.get(index, '[UNK]')


    def convert_tokens_to_string(self, tokens):
        return ' '.join(f'[{token}]' if token.isdigit() else token for token in tokens)


    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        # Add special tokens (CLS, SEP, etc.)
        return [self.vocab['[CLS]']] + token_ids_0 + [self.vocab['[SEP]']]

    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
        return [1] + ([0] * len(token_ids_0)) + [1]

    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
        return len(token_ids_0) * [0]

    def save_vocabulary(self, save_directory, filename_prefix=None):
        vocab_file = f"{save_directory}/{filename_prefix}_vocab.json"
        with open(vocab_file, 'w') as f:
            json.dump(self.vocab, f)
        return (vocab_file,)


if __name__ == '__main__':
    # 创建词汇表
    vocab = {f'[{str(i)}]': i for i in range(65536)}
    vocab['[CLS]'] = 65536
    vocab['[SEP]'] = 65537
    vocab['[UNK]'] = 65538

    # 保存词汇表到文件
    with open('/huyuqi/xmyu/VQProteinFormer/GLM/vocab.json', 'w') as f:
        json.dump(vocab, f)

    # 加载自定义 tokenizer
    tokenizer = PDBTokenizer(vocab_file='/huyuqi/xmyu/VQProteinFormer/GLM/vocab.json')

    # 测试 tokenizer
    text = "[1] [2] [65520] [MASK] [SEP]"
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    print("Tokens:", tokens)
    print("Token IDs:", token_ids)