from torch.nn.modules.loss import _Loss
from torch.autograd import Variable
import torch
from torch.functional import F

class DiscriminativeLoss(_Loss):

    def __init__(self, delta_var=0.5, delta_dist=1.5,
                 norm=2, alpha=1.0, beta=1.0, gamma=0.001,
                 device_cuda = False):
        super(DiscriminativeLoss, self).__init__(reduction='mean')
        self.delta_var = delta_var
        self.delta_dist = delta_dist
        self.norm = norm
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.device = device_cuda
        assert self.norm in [1, 2]

    def forward(self, input, target, n_clusters):
        # _assert_no_grad(target)
        # input是lanenet中的decode_logit,2个通道
        # target是分类为n个channel的ground_truth
        # n_clusters就是一个batchsize长度的列表，每个元素都是分类数, [5,5,5,5,5...]
        return self._discriminative_loss(input, target, n_clusters)

    def _discriminative_loss(self, input, target, n_clusters):
        bs, n_features, height, width = input.size()
        # target的channel数量即表示分类数量和聚类数量
        max_n_clusters = target.size(1)
        # 整理维度，bacth_size, channel ,height*width
        input = input.contiguous().view(bs, n_features, height * width)
        target = target.contiguous().view(bs, max_n_clusters, height * width)
        # 计算不同的损失
        c_means = self._cluster_means(input, target, n_clusters)
        l_var = self._variance_term(input, target, c_means, n_clusters)
        l_dist = self._distance_term(c_means, n_clusters)
        l_reg = self._regularization_term(c_means, n_clusters)
        loss = self.alpha * l_var + self.beta * l_dist + self.gamma * l_reg

        return loss, l_var, l_dist, l_reg

    def _cluster_means(self, input, target, n_clusters):
        bs, n_features, n_loc = input.size()
        max_n_clusters = target.size(1)

        # 在维度2的地方增加一个维度，变成bs, n_features,1, n_loc
        # expand将最后一个维度扩展，变成max_n_clusters, n_loc，
        # 最后变成bs, n_features,max_n_clusters, n_loc
        input = input.unsqueeze(2).expand(bs, n_features, max_n_clusters, n_loc)
        # bs, 1, max_n_clusters, n_loc
        target = target.unsqueeze(1)
        # 相乘之后变成bs, n_features, max_n_clusters, n_loc
        # 即将n个通道的ground_truth分别与网络输出的decode_logit相乘
        input = input * target

        means = []
        # 对于每个batch内的样本
        for i in range(bs):
            # n_features, n_clusters, n_loc
            input_sample = input[i, :, :n_clusters[i]]
            # 1, n_clusters, n_loc,
            target_sample = target[i, :, :n_clusters[i]]
            # 对第三个维度求和，除以target的第三个维度的求和，输出n_features, n_cluster
            mean_sample = input_sample.sum(2) / (target_sample.sum(2)+1E-10)
            # 如果分子分母都为0，输出NaN，变成0
            mean_sample[torch.isnan(mean_sample)] = 0

            # # 如果最大分类数大于本次输入的分类数，需要padding
            # n_pad_clusters = max_n_clusters - n_clusters[i]
            # assert n_pad_clusters >= 0
            # if n_pad_clusters > 0:
            #     pad_sample = torch.zeros(n_features, n_pad_clusters)
            #     pad_sample = Variable(pad_sample)
            #     mean_sample = torch.cat((mean_sample, pad_sample), dim=1)
            means.append(mean_sample)

        # bs, n_features, max_n_clusters
        means = torch.stack(means)

        return means

    def _variance_term(self, input, target, c_means, n_clusters):
        bs, n_features, n_loc = input.size()
        max_n_clusters = target.size(1)

        # bs, n_features, max_n_clusters, n_loc
        c_means = c_means.unsqueeze(3).expand(bs, n_features, max_n_clusters, n_loc)
        # bs, n_features, max_n_clusters, n_loc
        input = input.unsqueeze(2).expand(bs, n_features, max_n_clusters, n_loc)
        # bs, max_n_clusters, n_loc，限制最小值为0
        # torch中平方不能用**2，无法在GPU中计算，
        var = torch.pow(torch.clamp(torch.norm((input - c_means), self.norm, 1) -
                           self.delta_var, min=0),2) * target

        var_term = 0
        for i in range(bs):
            # 将每个batch的样本取出来
            # n_clusters, n_loc
            var_sample = var[i, :n_clusters[i]]
            # n_clusters, n_loc
            target_sample = target[i, :n_clusters[i]]

            # n_clusters，将维度1的值求和后相除
            # 对于少于4根车道的图像来说，最后几位的值为0，因此c_var的最后有0/0
            c_var = var_sample.sum(1) / (target_sample.sum(1)+1E-10)
            var_term = var_term + c_var.sum() / n_clusters[i]
        var_term = var_term/bs

        return var_term

    def _distance_term(self, c_means, n_clusters):
        bs, n_features, max_n_clusters = c_means.size()

        dist_term = 0
        for i in range(bs):
            if n_clusters[i] <= 1:
                continue

            # n_features, n_clusters
            mean_sample = c_means[i, :, :n_clusters[i]]

            # n_features, n_clusters, n_clusters
            means_a = mean_sample.unsqueeze(2).expand(n_features, n_clusters[i], n_clusters[i])
            # 维度不变，交换数据，相当于转置
            means_b = means_a.permute(0, 2, 1)
            diff = means_a - means_b

            # 该项中的参数均未在gpu中，因此需要判断在cpu还是gpu中计算
            margin = 2 * self.delta_dist * (1.0 - torch.eye(n_clusters[i]))
            if self.device:
                margin = Variable(margin).cuda()
                
            c_dist = torch.sum(torch.pow(torch.clamp(margin - torch.norm(diff, self.norm, 0), min=0) ,2))
            dist_term = dist_term + c_dist / (2 * n_clusters[i] * (n_clusters[i] - 1))
        dist_term = dist_term /bs

        return dist_term

    def _regularization_term(self, c_means, n_clusters):
        bs, n_features, max_n_clusters = c_means.size()

        reg_term = 0
        for i in range(bs):
            # n_features, n_clusters
            mean_sample = c_means[i, :, :n_clusters[i]]
            reg_term = reg_term + torch.mean(torch.norm(mean_sample, self.norm, 0))
        reg_term = reg_term / bs

        return reg_term


if __name__ == '__main__':
    input = torch.rand((4,5,512,256)).cuda()
    target = torch.randint(0,3,(4,512,256))
    target_one_hot = F.one_hot(target)
    target_label = target_one_hot.permute([0,3,1,2])
    print(target_label.shape)
    padding = torch.zeros((4,2,512,256)).type(torch.LongTensor)
    target = torch.cat((target_label,padding),dim=1).cuda()
    print(target.shape)

    print(target.shape)
    loss = DiscriminativeLoss(device_cuda = True)
    result = loss(input,target,[5,5,5,5])
    print(result)