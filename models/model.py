class CNN(nn.Module):
    def __init__(self, channels_in):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(channels_in, 64, 3, 1, 1, padding_mode='reflect')
        self.norm = nn.LayerNorm(64)
        self.mha = nn.MultiheadAttention(64, num_heads=1, batch_first=True)
        self.scale = nn.Parameter(torch.zeros(1))
        self.conv2 = nn.Conv2d(64, 64, 3, 2, 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, 2, 1)
        self.bn3 = nn.BatchNorm2d(128)
        self.do = nn.Dropout(0.5)
        self.fc_out = nn.Linear(128*4*4, 10)

    def use_attention(self, x):
        bs, c, h, w = x.shape
        x_att = x.reshape(bs, c, h * w).transpose(1, 2)
        x_att = self.norm(x_att)
        att_out, att_map = self.mha(x_att, x_att, x_att)
        return att_out.transpose(1, 2).reshape(bs, c, h, w), att_map

    def forward(self, x):
        x = self.conv1(x)
        x = self.scale * self.use_attention(x)[0] + x
        x = F.relu(x)
        x = F.relu(self.bn1(self.conv2(x)))
        x = F.relu(self.bn2(self.conv3(x)))
        x = F.relu(self.bn3(self.conv4(x)))
        x = self.do(x.reshape(x.shape[0], -1))
        return self.fc_out(x)