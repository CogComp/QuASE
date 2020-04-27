import torch
from torch.autograd import Variable


class SpanRepAssembly(torch.nn.Module):
    def __init__(self):
        super(SpanRepAssembly, self).__init__()

    def forward(self,
            hiddenA,
            hiddenB,
            maskA: torch.Tensor,
            maskB: torch.Tensor):

        combined_left, combined_right, mask = cross_product_combine(hiddenA, hiddenB, maskA, maskB, ordered=False)

        return combined_left, combined_right, mask


def cross_product_combine(hiddenA, hiddenB, maskA, maskB, ordered=False):
        batchA, timeA, embsizeA = hiddenA.size()
        batchB, timeB, embsizeB = hiddenB.size()

        assert batchA == batchB
        assert embsizeA == embsizeB

        if ordered:
            assert timeA == timeB
            out_num = int((timeA * (timeA+1)) / 2)

            hiddenA_data = hiddenA.data if isinstance(hiddenA, Variable) else hiddenA
            indexA = hiddenA_data.new().long().resize_(out_num).copy_(torch.Tensor([start for start in range(timeA) for i in range(start, timeA)]))
            indexA = Variable(indexA) if isinstance(hiddenA, Variable) else indexA
            hiddenA_rep = hiddenA.index_select(1, indexA)
            maskA_rep = maskA.index_select(1, indexA)

            indexB = hiddenA_data.new().long().resize_(out_num).copy_(torch.Tensor([i for start in range(timeA) for i in range(start, timeA)]))
            indexB = Variable(indexB) if isinstance(hiddenB, Variable) else indexB
            hiddenB_rep = hiddenB.index_select(1, indexB)
            maskB_rep = maskB.index_select(1, indexB)
        else:
            hiddenA_rep = hiddenA.view(batchA, timeA, 1, embsizeA)
            hiddenA_zero = Variable(torch.zeros(batchA, timeA, 1, embsizeA).cuda())
            hiddenB_rep = hiddenB.view(batchB, 1, timeB, embsizeB)
            hiddenB_zero = Variable(torch.zeros(batchB, 1, timeB, embsizeB).cuda())
            maskA_rep = maskA.view(batchA, timeA, 1)
            maskB_rep = maskB.view(batchB, 1, timeB)

        combined_left = (hiddenA_rep + hiddenB_zero).view(batchA, -1, embsizeA)
        combined_right = (hiddenB_rep + hiddenA_zero).view(batchA, -1, embsizeB)
        # combined = torch.cat((combined_left, combined_right), dim=-1)
        # combined = (hiddenA_rep + hiddenB_rep).view(batchA, -1, embsizeA)
        mask = (maskA_rep * maskB_rep).view(batchA, -1)

        return combined_left, combined_right, mask



 
