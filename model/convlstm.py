import torch
import torch.nn as nn


def frame_to_stream(in_list):
    temp_state = [temp.unsqueeze(2) for temp in in_list]
    cat_result = torch.cat(temp_state, dim=2)
    return cat_result


class ConvLSTMCell(nn.Module):
    def __init__(
        self,
        input_channels,
        hidden_channels,
        in_b_n_h_w,
        kernel_size=3,
        stride=1,
        size_flag="remain",
    ):
        super(ConvLSTMCell, self).__init__()
        padding = kernel_size // 2
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.stride = stride
        try:
            b, c, n, h, w = in_b_n_h_w
            in_dim_num = 5
        except:
            b, n, w = in_b_n_h_w
            in_dim_num = 3
        self.size_flag = size_flag
        if in_dim_num == 5:
            self.wci = torch.zeros(
                b, hidden_channels, h, w, requires_grad=True, device="cuda"
            )
            self.wcf = torch.zeros(
                b, hidden_channels, h, w, requires_grad=True, device="cuda"
            )
            self.wco = torch.zeros(
                b, hidden_channels, h, w, requires_grad=True, device="cuda"
            )
        elif in_dim_num == 3:
            self.wci = torch.zeros(
                b, hidden_channels, w, requires_grad=True, device="cuda"
            )
            self.wcf = torch.zeros(
                b, hidden_channels, w, requires_grad=True, device="cuda"
            )
            self.wco = torch.zeros(
                b, hidden_channels, w, requires_grad=True, device="cuda"
            )
        if self.size_flag == "remain":
            self.conv = nn.Conv2d(
                in_channels=input_channels + hidden_channels,
                out_channels=4 * hidden_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
        elif self.size_flag == "up":
            self.conv = nn.Conv2d(
                in_channels=input_channels + input_channels,
                out_channels=4 * hidden_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )

    def forward(self, input_tensor, cur_state, first_flag):
        if first_flag:
            h_cur, c_cur = cur_state[-1]
        else:
            h_cur, c_cur = cur_state
        combined = torch.cat((input_tensor, h_cur), dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(
            combined_conv, combined_conv.size(1) // 4, dim=1
        )  # [8, 128, 28, 28]
        if self.size_flag == "up":
            c_cur = reshape_tensor(c_cur, cc_i)
            self.wci = reshape_tensor(self.wci, cc_i)
            self.wcf = reshape_tensor(self.wcf, cc_f)
            self.wco = reshape_tensor(self.wco, cc_o)

        i = torch.sigmoid(
            cc_i + self.wci[:, :, : c_cur.size(2), : c_cur.size(2)] * c_cur
        )
        f = torch.sigmoid(
            cc_f + self.wcf[:, :, : c_cur.size(2), : c_cur.size(2)] * c_cur
        )
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        o = torch.sigmoid(
            cc_o + self.wco[:, :, : c_next.size(2), : c_cur.size(2)] * c_next
        )
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


def reshape_tensor(source_tensor, target_tensor):
    t_b, t_c, t_h, t_w = target_tensor.shape
    s_b, s_c, s_h, s_w = source_tensor.shape
    temp_tensor = source_tensor.view(t_b, -1, t_h, t_w)
    if temp_tensor.size(1) > t_c:
        temp_tensor = temp_tensor[:, :t_c, :, :]
    elif temp_tensor.size(1) / t_c == 2:
        temp_tensor = torch.cat((temp_tensor, temp_tensor), dim=1)
    else:
        temp_tensor = torch.cat((temp_tensor, temp_tensor), dim=1)
        temp_tensor = temp_tensor[:, :t_c, :, :]
    return temp_tensor


class ConvLSTM(nn.Module):
    def __init__(
        self,
        input_channels,
        hidden_channels,
        in_b_n_h_w=(8, 3, 64, 32, 32),
        kernel_size=3,
        stride=1,
        padding=1,
        num_layers=1,
        first_flag=False,
        size_flag="remain",
        input_mode="stream",
    ):
        super(ConvLSTM, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.num_layers = num_layers
        self.size_flag = size_flag
        self.input_mode = input_mode
        if self.stride == 2 or self.size_flag == "up":
            self.stride = 2
            self.size_flag = "up"
        try:
            b, c, n, h, w = in_b_n_h_w
            self.in_dim_num = 5
        except:
            b, n, w = in_b_n_h_w
            self.in_dim_num = 3

        self.first_flag = first_flag
        cell_list = []
        for i in range(self.num_layers):
            cur_input_channels = input_channels if i == 0 else hidden_channels
            cell_list.append(
                ConvLSTMCell(
                    cur_input_channels,
                    hidden_channels,
                    in_b_n_h_w,
                    kernel_size,
                    stride,
                    size_flag=self.size_flag,
                )
            )
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, c_state=None):
        if self.input_mode == "stream":
            in_b_c_h_w = list(input_tensor[-1].shape)
            if isinstance(input_tensor, torch.Tensor):
                seq_len = in_b_c_h_w[1]
            elif isinstance(input_tensor, list):
                seq_len = len(input_tensor)
        elif self.input_mode == "frame":
            in_b_c_h_w = list(input_tensor.shape)
            seq_len = 1

        if c_state is None:
            c_state = self.init_hidden(
                input_tensor, in_b_c_h_w, in_mode_flag=self.input_mode
            )
        h_states = []
        c_states = []

        for seq in range(seq_len):
            if self.input_mode == "stream":
                if isinstance(input_tensor, torch.Tensor):
                    temp_input = input_tensor[:, :, seq, :, :]
                elif isinstance(input_tensor, list):
                    temp_input = input_tensor[seq]
            elif self.input_mode == "frame":
                temp_input = input_tensor

            for i in range(self.num_layers):
                if c_state is None or c_state == []:
                    cur_hidden = None
                # elif self.input_mode == 'stream':
                elif self.input_mode == "frame" and not self.first_flag:
                    cur_hidden = c_state
                else:
                    cur_hidden = c_state[i]

                h_state, cell_state = self.cell_list[i](
                    temp_input, (temp_input, cur_hidden), self.first_flag
                )
                if self.input_mode == "frame":
                    h_states, c_states = h_state, cell_state
                elif self.input_mode == "stream":
                    h_states.append(h_state)
                    c_states.append(cell_state)
        return h_states, c_states

    def init_hidden(
        self, input_tensor, in_b_c_h_w, device="cuda", in_mode_flag="stream"
    ):
        hidden_states = []
        for i in range(self.num_layers):
            if i == 0:
                if len(in_b_c_h_w) == 5:
                    hidden_states.append(
                        (
                            torch.zeros(
                                in_b_c_h_w[0],
                                in_b_c_h_w[1],
                                in_b_c_h_w[3],
                                in_b_c_h_w[4],
                                device=device,
                            ),
                            torch.zeros(
                                in_b_c_h_w[0],
                                in_b_c_h_w[1],
                                in_b_c_h_w[3],
                                in_b_c_h_w[4],
                                device=device,
                            ),
                        )
                    )
                elif len(in_b_c_h_w) == 4 and in_mode_flag == "frame":
                    hidden_states.append(
                        (
                            torch.zeros(*in_b_c_h_w, device=device),
                            torch.zeros(*in_b_c_h_w, device=device),
                        )
                    )
                elif len(in_b_c_h_w):
                    hidden_states.append(
                        (
                            torch.zeros(
                                input_tensor.shape[0],
                                in_b_c_h_w[1],
                                in_b_c_h_w[2],
                                in_b_c_h_w[3],
                                device=device,
                            ),
                            torch.zeros(
                                input_tensor.shape[0],
                                in_b_c_h_w[1],
                                in_b_c_h_w[2],
                                in_b_c_h_w[3],
                                device=device,
                            ),
                        )
                    )
            else:
                hidden_states.append(None)
        return hidden_states


def init_weights(modules, device="cuda"):
    """
    Weight initialization from original SensorFusion Code
    """
    for m in modules:
        if (
            isinstance(m, nn.Conv2d)
            or isinstance(m, nn.ConvTranspose2d)
            or isinstance(m, nn.Conv1d)
        ):
            nn.init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
