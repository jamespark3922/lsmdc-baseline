import torch
import misc.utils as utils

# transfer model1 visual embeddings to model2
def transfer_visual_embeddings(model1, model2):
    with torch.no_grad():
        v2 = model2.get_visual_embeddings()
        for i,v in enumerate(model1.get_visual_embeddings()):
            v2[i].weight.data = v.weight.data.clone()

def train_generator(gen_model, gen_optimizer, crit, loader, grad_clip=0.1):
    gen_model.train()
    data = loader.get_batch('train')
    torch.cuda.synchronize()
    tmp = [data['fc_feats'], data['img_feats'], data['box_feats'],
           data['labels'], data['masks']]
    tmp = [_ if _ is None else torch.from_numpy(_).cuda() for _ in tmp]
    fc_feats, img_feats, box_feats, labels, masks = tmp

    sent_num = data['sent_num']
    wrapped = data['bounds']['wrapped']
    gen_optimizer.zero_grad()

    seq = gen_model(fc_feats, img_feats, box_feats, labels)
    seq = utils.align_seq(sent_num, seq)
    labels = utils.align_seq(sent_num, labels)
    masks = utils.align_seq(sent_num, masks)

    loss = crit(seq, labels[:,1:], masks[:,1:])
    loss.backward()
    gen_loss = loss.item()

    utils.clip_gradient(gen_optimizer, grad_clip)
    gen_optimizer.step()
    torch.cuda.synchronize()

    return gen_loss, wrapped, sent_num