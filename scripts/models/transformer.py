import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x, offset=0):
        """
        Args:
            x:      (batch, seq_len, d_model)
            offset: positional index offset for autoregressive decoding
        """
        x = x + self.pe[:, offset : offset + x.size(1), :]
        return self.dropout(x)


class TimeSeriesTransformer(nn.Module):
    """
    Encoder-decoder transformer for multi-step cloud cost forecasting.

    Training:  teacher-forced — decoder receives lagged cost in channel 0
    Inference: autoregressive via CostForecaster.predict()
    """

    def __init__(
        self,
        input_dim=9,
        d_model=48,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=128,
        dropout=0.2,
        output_dim=1,
    ):
        super().__init__()

        self.d_model   = d_model
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Separate projections: encoder sees full feature set,
        # decoder sees sparse cost-only input — different learned embeddings
        self.encoder_projection = nn.Linear(input_dim, d_model)
        self.decoder_projection = nn.Linear(input_dim, d_model)

        # Positional encoding with dropout
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,    # Pre-norm: more stable training on small datasets
        )

        # Output head: project d_model -> 1 (scaled log cost)
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _generate_causal_mask(self, sz, device):
        """Upper-triangular mask — prevents decoder attending to future steps."""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        return mask.masked_fill(mask == 1, float('-inf'))

    def forward(self, src, tgt):
        """
        Teacher-forced forward pass (used during training).

        Args:
            src: (batch, src_len, input_dim)  historical features
            tgt: (batch, tgt_len, input_dim)  decoder input, cost in channel 0

        Returns:
            (batch, tgt_len, output_dim)
        """
        src = self.pos_encoder(self.encoder_projection(src))
        tgt = self.pos_encoder(self.decoder_projection(tgt))

        tgt_mask = self._generate_causal_mask(tgt.size(1), tgt.device)

        output = self.transformer(src=src, tgt=tgt, tgt_mask=tgt_mask)
        return self.output_projection(output)

    def predict(self, src, forecast_length):
        """
        Autoregressive inference — no teacher forcing.

        Args:
            src:             (batch, src_len, input_dim) scaled features
            forecast_length: steps to predict

        Returns:
            (batch, forecast_length, output_dim) — in scaled log space
        """
        self.eval()
        with torch.no_grad():
            device = src.device

            # Encode source once
            src_encoded = self.pos_encoder(self.encoder_projection(src))
            memory      = self.transformer.encoder(src_encoded)

            batch_size = src.size(0)

            # Seed: zeros with last known cost in channel 0
            decoder_input = torch.zeros(batch_size, 1, self.input_dim, device=device)

            predictions = []
            for step in range(forecast_length):
                tgt      = self.pos_encoder(
                    self.decoder_projection(decoder_input), offset=0)
                tgt_mask = self._generate_causal_mask(tgt.size(1), device)

                dec_out  = self.transformer.decoder(
                    tgt=tgt, memory=memory, tgt_mask=tgt_mask)
                pred     = self.output_projection(dec_out[:, -1:, :])
                predictions.append(pred)

                # Next decoder token: predicted cost in channel 0, zeros elsewhere
                next_token          = torch.zeros(batch_size, 1, self.input_dim, device=device)
                next_token[:, :, 0] = pred.squeeze(-1)
                decoder_input       = torch.cat([decoder_input, next_token], dim=1)

            return torch.cat(predictions, dim=1)  # (batch, forecast_length, output_dim)


class CostForecaster:
    """
    High-level wrapper — handles device, save/load, and inference.
    """

    def __init__(
        self,
        input_dim=9,
        d_model=48,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=128,
        dropout=0.2,
        output_dim=1,
        device='cpu',
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        self.model = TimeSeriesTransformer(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            output_dim=output_dim,
        ).to(self.device)

    def save(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved -> {path}")

    def load(self, path):
        self.model.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=True)
        )
        print(f"Model loaded <- {path}")

    def predict(self, historical_data, last_known_cost_scaled, forecast_hours=24):
        """
        Run autoregressive inference.

        Args:
            historical_data:        numpy (batch, seq_len, input_dim) feature-scaled
            last_known_cost_scaled: numpy (batch,) last known cost in scaled log space
                                    i.e. cost_scaler.transform(np.log1p([[cost]]))
            forecast_hours:         steps to forecast

        Returns:
            numpy (batch, forecast_hours, 1) in scaled log space.
            To get real $: np.expm1(cost_scaler.inverse_transform(result))
        """
        self.model.eval()

        src    = torch.FloatTensor(historical_data).to(self.device)
        memory = self.model.transformer.encoder(
            self.model.pos_encoder(self.model.encoder_projection(src))
        )

        batch_size    = src.size(0)
        decoder_input = torch.zeros(
            batch_size, 1, self.model.input_dim, device=self.device)
        decoder_input[:, 0, 0] = torch.FloatTensor(
            last_known_cost_scaled).to(self.device)

        predictions = []
        with torch.no_grad():
            for _ in range(forecast_hours):
                tgt      = self.model.pos_encoder(
                    self.model.decoder_projection(decoder_input), offset=0)
                tgt_mask = self.model._generate_causal_mask(
                    tgt.size(1), self.device)

                dec_out  = self.model.transformer.decoder(
                    tgt=tgt, memory=memory, tgt_mask=tgt_mask)
                pred     = self.model.output_projection(dec_out[:, -1:, :])
                predictions.append(pred)

                next_token          = torch.zeros(
                    batch_size, 1, self.model.input_dim, device=self.device)
                next_token[:, :, 0] = pred.squeeze(-1)
                decoder_input       = torch.cat([decoder_input, next_token], dim=1)

        return torch.cat(predictions, dim=1).cpu().numpy()