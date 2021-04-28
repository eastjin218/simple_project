from torchtext import data

class DataLoader(object):

    def __init__(
        self, train_fn,
        batch_size=64,
        valid_radio=.2,
        device=-1,
        max_vocab=9999999,
        min_freq=1,
        use_eos=False,
        shuffle=True,
    ):

        super().__init__()

        self.

        self.text= data.Field(
            use_vocab = True,
            batch_first =True,
            include_lengths =False,
            eos_token='<EOS>' if use_eos else None,
        )

