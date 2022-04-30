import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from loader import data_loader
from models import VQAModel

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    data_loader = {
        'train': get_loader(
            input_dir=args.input_dir,
            input_vqa='train.npy',
            max_qst_length=args.max_question_length,
            max_num_ans=args.max_num_ans,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        ),
        'valid': get_loader(
            input_dir=args.input_dir,
            input_vqa='valid.npy',
            max_qst_length=args.max_question_length,
            max_num_ans=args.max_num_ans,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
    }

    question_vocabulary_size = data_loader['train'].dataset.question_vocabulary.vocabulary_size
    ans_vocabulary_size = data_loader['train'].dataset.ans_vocabulary.vocabulary_size

    model = VQAModel(
        embed_size=args.embed_size,
        question_vocab_size=question_vocabulary_size,
        ans_vocab_size=ans_vocabulary_size,
        word_embed_size=args.word_embed_size,
        num_layers=args.num_layers,
        hidden_size=args.hidden_layer_size).to(device)

    if args.load_step!=0:
        model.load_state_dict(torch.load(os.path.join(args.model_dir, 'model-epoch-{:02d}.ckpt'.format(args.load_step))))

    params = list(model.image_encoder.fc.parameters()) \
        + list(model.question_encoder.parameters()) \
        + list(model.fc1.parameters()) \
        + list(model.fc2.parameters())

    optimizer = optim.Adam(params, lr=args.learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.load_model,args.num_epochs):
        for phase in ['train','valid']:
            running_loss = 0.0
            running_corr_exp = 0
            batch_step_size = len(data_loader[phase].dataset) / args.batch_size

            if phase == 'train':
                model.train()
            else:
                model.eval()

            for batch_idx, batch_sample in enumerate(data_loader[phase]):

                image = batch_sample['image'].to(device)
                question = batch_sample['question'].to(device)
                label = batch_sample['answer_label'].to(device)
                multi_choice = batch_sample['answer_multi_choice']  # not tensor, list.

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):

                    output=model(image,question)
                    _,pred_exp=torch.max(output,1)
                    loss=criterion(output,label)

                    loss.backward()
                    optimizer.step()

                running_loss+=loss.item()
                running_corr_exp+=torch.stack([(ans == pred_exp.cpu()) for ans in multi_choice]).any(dim=0).sum()

                # Print the average loss in a mini-batch.
                if batch_idx % 100 == 0:
                    print('| {} SET | Epoch [{:02d}/{:02d}], Step [{:04d}/{:04d}], Loss: {:.4f}'
                          .format(phase.upper(), epoch+1, args.num_epochs, batch_idx, int(batch_step_size), loss.item()))

            if (epoch+1) % args.save_step == 0:
                torch.save({'epoch': epoch+1, 'state_dict': model.state_dict()},
                       os.path.join(args.model_dir, 'model-epoch-{:02d}.ckpt'.format(epoch+1)))


    

if __name__=='main':

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', type=str, default='./datasets',
                        help='input directory for visual question answering.')

    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='directory for logs.')

    parser.add_argument('--model_dir', type=str, default='./models',
                        help='directory for saved models.')

    parser.add_argument('--max_qustion_length', type=int, default=30,
                        help='maximum length of question. \
                              the length in the VQA dataset = 26.')

    parser.add_argument('--max_num_ans', type=int, default=10,
                        help='maximum number of answers.')

    parser.add_argument('--embed_size', type=int, default=1024,
                        help='embedding size of feature vector \
                              for both image and question.')

    parser.add_argument('--word_embed_size', type=int, default=300,
                        help='embedding size of word \
                              used for the input in the LSTM.')

    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers of the RNN(LSTM).')

    parser.add_argument('--hidden_layer_size', type=int, default=512,
                        help='hidden layer size in the LSTM.')

    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate for training.')

    parser.add_argument('--step_size', type=int, default=10,
                        help='period of learning rate decay.')

    parser.add_argument('--gamma', type=float, default=0.1,
                        help='multiplicative factor of learning rate decay.')

    parser.add_argument('--num_epochs', type=int, default=30,
                        help='number of epochs.')

    parser.add_argument('--batch_size', type=int, default=80,
                        help='batch_size.')

    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of processes working on cpu.')

    parser.add_argument('--save_step', type=int, default=1,
                        help='save step of model.')

    parser.add_argument('--load_step', type=int, default=0,
                        help='load saved model.')

    args = parser.parse_args()

    main(args)