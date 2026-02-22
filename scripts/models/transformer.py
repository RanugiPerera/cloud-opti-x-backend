import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Adds positional information to input embeddings
    Helps the model understand sequence order
    """

    def __init__(self, d_model, max_len=5000):
        super().__init__()

        # create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # add batch dimension

        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]


class TimeSeriesTransformer(nn.Module):
    """
    Transformer model for cloud cost forecasting

    Architecture:
    - Input: historical cost data (past N days)
    - Output: predicted costs (future M days)
    """

    def __init__(
            self,
            input_dim=4,  # features: cpu, memory, duration, cost
            d_model=64,  # embedding dimension
            nhead=4,  # number of attention heads
            num_encoder_layers=3,  # transformer encoder layers
            num_decoder_layers=3,  # transformer decoder layers
            dim_feedforward=256,  # feedforward network dimension
            dropout=0.1,
            output_dim=1  # predict cost only
    ):
        super().__init__()

        self.d_model = d_model
        self.input_dim = input_dim
        self.output_dim = output_dim

        # input projection layer
        self.input_projection = nn.Linear(input_dim, d_model)

        # positional encoding
        self.pos_encoder = PositionalEncoding(d_model)

        # transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        # output projection layer
        self.output_projection = nn.Linear(d_model, output_dim)

        # initialize weights
        self._init_weights()

    def _init_weights(self):
        # xavier initialization for better training
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt):
        """
        Forward pass

        Args:
            src: source sequence (historical data) - shape: (batch, src_len, input_dim)
            tgt: target sequence (future data) - shape: (batch, tgt_len, input_dim)

        Returns:
            predictions: shape (batch, tgt_len, output_dim)
        """
        # project input to model dimension
        src = self.input_projection(src)  # (batch, src_len, d_model)
        tgt = self.input_projection(tgt)  # (batch, tgt_len, d_model)

        # add positional encoding
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)

        # create causal mask for decoder (prevent looking into future)
        tgt_mask = self._generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)

        # transformer forward pass
        output = self.transformer(
            src=src,
            tgt=tgt,
            tgt_mask=tgt_mask
        )

        # project to output dimension
        predictions = self.output_projection(output)

        return predictions

    def _generate_square_subsequent_mask(self, sz):
        """
        Generate causal mask to prevent attention to future positions
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def predict(self, src, forecast_length):
        """
        Predict future values autoregressively

        Args:
            src: historical data - shape: (batch, src_len, input_dim)
            forecast_length: number of steps to predict

        Returns:
            predictions: shape (batch, forecast_length, output_dim)
        """
        self.eval()
        with torch.no_grad():
            batch_size = src.size(0)
            device = src.device

            # encode source
            src_encoded = self.pos_encoder(self.input_projection(src))

            # initialize decoder input with last value from source
            # shape: (batch, 1, input_dim)
            decoder_input = src[:, -1:, :]

            predictions = []

            # generate predictions autoregressively
            for _ in range(forecast_length):
                # prepare decoder input
                tgt = self.input_projection(decoder_input)
                tgt = self.pos_encoder(tgt)

                # create mask
                tgt_mask = self._generate_square_subsequent_mask(tgt.size(1)).to(device)

                # transformer forward
                output = self.transformer.decoder(
                    tgt=tgt,
                    memory=src_encoded,
                    tgt_mask=tgt_mask
                )

                # project to prediction
                pred = self.output_projection(output[:, -1:, :])  # (batch, 1, output_dim)
                predictions.append(pred)

                # update decoder input
                # create next input by combining prediction with other features
                # (for simplicity, we'll repeat the last known feature values)
                next_input = decoder_input[:, -1:, :].clone()
                next_input[:, :, -1] = pred.squeeze(-1)  # update cost feature

                # append to decoder input
                decoder_input = torch.cat([decoder_input, next_input], dim=1)

            # stack all predictions
            predictions = torch.cat(predictions, dim=1)  # (batch, forecast_length, output_dim)

        return predictions


# simple model wrapper for easier use
class CostForecaster:
    """
    High-level wrapper for the Transformer model
    Makes training and prediction easier
    """

    def __init__(
            self,
            input_dim=4,
            d_model=64,
            nhead=4,
            num_encoder_layers=3,
            num_decoder_layers=3,
            dim_feedforward=256,
            dropout=0.1,
            output_dim=1,
            device='cpu'
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
            output_dim=output_dim
        ).to(self.device)

    def save(self, path):
        """Save model weights"""
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path):
        """Load model weights"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Model loaded from {path}")

    def predict(self, historical_data, forecast_hours=7):
        """
        Make predictions

        Args:
            historical_data: numpy array of shape (batch, seq_len, features)
            forecast_hours: number of hours to forecast

        Returns:
            predictions: numpy array of shape (batch, forecast_hours)
        """
        self.model.eval()

        # convert to tensor
        src = torch.FloatTensor(historical_data).to(self.device)

        # predict
        predictions = self.model.predict(src, forecast_hours)

        # convert back to numpy
        return predictions.cpu().numpy()