import os
import cv2
import time
import json
import argparse

from visualize          import *
from prepare.processor  import *
from loss.contrastive   import *
from loss.variational   import *
from models.vae         import VariationalAutoencoder

def read_tsv(path):
    datas = []
    with open(path, 'r', encoding='utf-8') as frs:
        for fr in tqdm(frs):
            data = fr.replace('\n', '')
            data = data.split('\t')
            datas.append(data)
    return datas

def write_json(path, datas):
    with open(path, 'w', encoding='utf-8') as fr:
        json.dump(datas, fr, indent=4)

def debug_model(
        real_boundary,
        latent_boundary,
        samples,
        model,
        dataloader,
        device
    ):
    model.eval()

    use_vae = model.use_vae
    model.use_vae = False
    real_result = predict_draw(
        x_max=real_boundary, 
        y_max=real_boundary, 
        samples=samples, 
        model=model,
        dataloader=dataloader,
        device=device,
        transform_q=None, 
        transform_d=None, 
    )
    model.use_vae = use_vae

    latent_result = predict_draw(
        x_max=latent_boundary, 
        y_max=latent_boundary, 
        samples=samples, 
        model=model.decoder,
        dataloader=dataloader,
        device=device,
        transform_q=model.encoder, 
        transform_d=model.encoder,
        zoom_auto=True
    )

    return real_result, latent_result

def train_epoch(
        model, 
        device, 
        dataloader,
        dataloader_doc_first, 
        optimizer,
        alpha,
        beta
    ):
    model.train()
    train_loss = 0.0
    # for queries, docs in dataloader:
    for queries, docs in dataloader_doc_first:
        q_x, d_x, c = process_fn(queries, docs, device)
        # forward model
        d_out = model(d_x)
        q_out = model(q_x)
        
        if model.use_vae:
            # loss_cl    = contrastive_loss(q_out.enc_out.latent, d_out.enc_out.latent, device)
            loss_cl_ga = contrastive_loss(c, q_out.enc_out.mean, d_out.enc_out.mean, d_out.enc_out.sigma, device, 'gaussian')
            loss_ce_d  = reconstruction_loss(d_out.logits, c)
            # loss_cl    = contrastive_loss(c, q_out.enc_out.mean, d_out.enc_out.mean, device)
            # loss_vae_d = variational_loss(d_out.logits, c, d_out.enc_out.mean, d_out.enc_out.sigma, beta=beta)
            # loss_vae_q = variational_loss(q_out.logits, c, q_out.enc_out.mean, q_out.enc_out.sigma, beta=beta)
            # loss       = alpha * (0.8 * loss_vae_d + 0.2 * loss_vae_q) + (1 - alpha) * loss_cl
            # loss  = alpha * loss_cl + (1 - alpha) * loss_cl_ga
            loss  =  alpha * loss_ce_d + (1 - alpha) * loss_cl_ga 
            # loss       = loss_vae_d + loss_vae_q
            # loss       = alpha * (loss_vae_d) + (1 - alpha) * loss_cl
        else:
            # loss_cl   = contrastive_loss(q_out.enc_out.mean, d_out.enc_out.mean, device)
            loss_ce_d = reconstruction_loss(d_out.logits, c)
            # loss = alpha * (loss_ce_d + loss_ce_q) + (1 - alpha) * loss_cl
            loss = alpha * (loss_ce_d) + (1 - alpha) * loss_cl
            # loss = loss_ce_d + loss_ce_q
            # loss = loss_ce_q
            # loss = loss_ce_d
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(dataloader.dataset)

def train(
        model, 
        epochs, 
        device, 
        dataloader,
        dataloader_doc_first, 
        optimizer,
        alpha,
        beta,
        debug_mode,
        debug_path,
        debug_step=1000
    ):
    model.to(device)
    for epoch in range(epochs):
        train_loss = train_epoch(model, device, dataloader, dataloader_doc_first, optim, alpha, beta)
        print('\n EPOCH {}/{} \t train loss {:.5f} \t '.format(epoch + 1, epochs, train_loss))

        if (debug_mode == True) and (epoch % debug_step == 0 or epoch == (epochs - 1)):
            samples = 1000
            real_boundary   = 0.5
            latent_boundary = 100
            
            real_result, latent_result = debug_model(
                real_boundary=real_boundary,
                latent_boundary=latent_boundary,
                samples=samples,
                model=model,
                dataloader=dataloader,
                device=device
            )
            
            debug_real_result_path   = os.path.join(debug_path, f'real_predict_epoch_{epoch}.png')
            debug_latent_result_path = os.path.join(debug_path, f'latent_predict_epoch_{epoch}.png')
            cv2.imwrite(debug_real_result_path, real_result)
            cv2.imwrite(debug_latent_result_path, latent_result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train VAE.")
    parser.add_argument("--dataset_path"   , type=str,   default='../dataset/train.tsv')
    parser.add_argument("--output_dir"     , type=str,   default='./exp')
    parser.add_argument("--latent_dims"    , type=int,   default=2)
    parser.add_argument("--n_class"        , type=int,   default=10)
    parser.add_argument("--learning_rate"  , type=float, default=0.005)
    parser.add_argument("--epochs"         , type=int,   default=50000)
    parser.add_argument("--batch_size"     , type=int,   default=1024)
    parser.add_argument("--loss_type"      , type=str,   default="vae+cl")
    parser.add_argument("--use_vae"        , type=str,   default=True)
    parser.add_argument("--alpha"          , type=float, default=0.2)
    parser.add_argument("--beta"           , type=float, default=0.99999)
    parser.add_argument("--debug_mode"     , type=str,   default=True)
    args = parser.parse_args()
    
    train_datas = read_tsv(args.dataset_path)
    train_datas = normalize(train_datas)

    doc_datas   = train_datas[:10]
    query_datas = train_datas[10:]
    train_datas = preprocess_dataset(doc_datas, query_datas)
    train_doc_first_datas = preprocess_dataset_document_first(doc_datas, query_datas, repeat_doc_datas=True)

    train_loader = torch.utils.data.DataLoader(train_datas, batch_size=args.batch_size)
    train_doc_first_loader = torch.utils.data.DataLoader(
        train_doc_first_datas, 
        batch_size=args.batch_size, 
        collate_fn=collate_fn
    )

    model     = VariationalAutoencoder(latent_dims=args.latent_dims, n_class=args.n_class)
    optim     = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Selected device: {device}')

    # create experiment env
    time_now = time.strftime("%Y_%d_%m__%H_%M_%S", time.localtime())
    exp_dir  = os.path.join(args.output_dir, time_now)
    os.mkdir(exp_dir)
    config_path = os.path.join(exp_dir, 'config.json')
    write_json(config_path, args.__dict__)
    debug_path  = os.path.join(exp_dir, 'debug')
    if args.debug_mode:
        os.mkdir(debug_path)

    # model training
    model.use_vae = args.use_vae
    train(
        model=model, 
        epochs=args.epochs, 
        device=device, 
        dataloader=train_loader, 
        dataloader_doc_first=train_doc_first_loader, 
        optimizer=optim,
        alpha=args.alpha,
        beta=args.beta,
        debug_mode=args.debug_mode,
        debug_path=debug_path,
        debug_step=100
    )