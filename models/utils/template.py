class ResData:
    def __init__(
        self, 
        logits  = None, 
        mean    = None, 
        sigma   = None, 
        latent  = None,
        enc_out = None
    ):
        self.logits  = logits
        self.mean    = mean
        self.sigma   = sigma
        self.latent  = latent
        self.enc_out = enc_out
    