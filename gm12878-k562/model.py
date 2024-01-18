import torch.nn as nn
import torch
# Neural Net definition
class DeepCBS_Network(nn.Module):
    def __init__(self):
        super(DeepCBS_Network, self).__init__()

        self.cnn_1 = nn.Sequential(
            nn.Conv1d(in_channels=100,
                      out_channels=200,
                      kernel_size=50,
                      stride=1,
                      padding=0),
            nn.ReLU(True),
            )
        self.cnn_2=nn.Sequential(
            nn.MaxPool1d(3,stride=2),

            nn.Conv1d(in_channels=200,
                      out_channels=128,
                      kernel_size=8,
                      stride=1,
                      padding=0),
            nn.ReLU(True),
            nn.MaxPool1d(3,stride=2),

            nn.Conv1d(in_channels=128,
                      out_channels=64,
                      kernel_size=4,
                      stride=1,
                      padding=0),
            nn.ReLU(True),
            nn.MaxPool1d(3,stride=2),
        )
        self.cnn_3 = nn.Sequential(
            nn.Conv1d(in_channels=100,
                      out_channels=200,
                      kernel_size=50,
                      stride=1,
                      padding=0),
            nn.ReLU(True),
            )
        self.cnn_4=nn.Sequential(
            nn.MaxPool1d(3,stride=2),

            nn.Conv1d(in_channels=200,
                      out_channels=128,
                      kernel_size=8,
                      stride=1,
                      padding=0),
            nn.ReLU(True),
            nn.MaxPool1d(3,stride=2),

            nn.Conv1d(in_channels=128,
                      out_channels=64,
                      kernel_size=4,
                      stride=1,
                      padding=0),
            nn.ReLU(True),
            nn.MaxPool1d(3,stride=2),
        )

        self.Bigru = nn.Sequential(
            nn.GRU(input_size=64,
                    hidden_size=128,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True,
                    bias=True)
        )

        self.Prediction = nn.Sequential(
            nn.Linear(256, 32),
            nn.Dropout(0.2),
            nn.Linear(32,1),
            nn.Sigmoid(),
        )

    def forward(self, input):
       # print(input.shape)
       # torch.Size([32, 250, 8])
        forward_input = input[:,:,:100]
       # torch.Size([5, 250, 8])
       # print(forward_input.shape)
        complementary_input=input[:,:,100:200]
        forward_input=forward_input.permute(0,2,1)
        complementary_input=complementary_input.permute(0,2,1)
        forward_out_1=self.cnn_1(forward_input)
        forward_out=self.cnn_2(forward_out_1)
        complementary_out_1=self.cnn_3(complementary_input)
        complementary_out=self.cnn_4(complementary_out_1)
        merge_data=torch.cat([forward_out,complementary_out],dim=2)
        merge_data=merge_data.permute(0,2,1)
        bilstm_out, _ = self.Bigru(merge_data)
        # print(bilstm_out.shape)
        bilstm_out = bilstm_out[:, -1, :]
        result = self.Prediction(bilstm_out)
        return result,forward_out_1,complementary_out_1
