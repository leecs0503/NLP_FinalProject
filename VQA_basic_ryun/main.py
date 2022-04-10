import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim # 최적화 알고리즘
from torch.optim import lr_scheduler # learning rate를 조절하는 scheduler
from data_loader import get_loader
from model import VqaModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    data_loader = get_loader(
        input_dir=args.input_dir,
        input_vqa_train='train.npy',
        input_vqa_valid='valid.npy',
        max_qst_length=args.max_qst_length,
        max_num_ans=args.max_num_ans,
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    qst_vocab_size = data_loader['train'].dataset.qst_vocab.vocab_size
    ans_vocab_size = data_loader['train'].dataset.ans_vocab.vocab_size
    ans_unk_idx = data_loader['train'].dataset.ans_vocab.unk2idx

    model = VqaModel(
        embed_size=args.embed_size,
        qst_vocab_size=qst_vocab_size,
        ans_vocab_size=ans_vocab_size,
        word_embed_size=args.word_embed_size,
        num_layers=args.num_layers,
        hidden_size=args.hidden_size).to(device)

    criterion = nn.CrossEntropyLoss()

    params = list(model.img_encoder.fc.parameters()) \
        + list(model.qst_encoder.parameters()) \
        + list(model.fc1.parameters()) \
        + list(model.fc2.parameters())

    optimizer = optim.Adam(params, lr=args.learning_rate)   # Adam optimizer 이용
    # 가장 흔히 사용되는 learning rate scheduler로 일정한 step마다 learning rate에 지정한 gamma를 곱해주어 learning rate를 감소시킨다.
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    for epoch in range(args.num_epochs):

        for phase in ['train', 'valid']:

            running_loss = 0.0
            running_corr_exp1 = 0   # 예측이 correct한 것들 저장하기 위함
            running_corr_exp2 = 0
            batch_step_size = len(data_loader[phase].dataset) / args.batch_size

            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()    # evaluation 과정에서 사용하지 않는 layer들 알아서 off

            for batch_idx, batch_sample in enumerate(data_loader[phase]):

                image = batch_sample['image'].to(device)
                question = batch_sample['question'].to(device)
                label = batch_sample['answer_label'].to(device)
                multi_choice = batch_sample['answer_multi_choice']  

                # pytorch에서는 gradient가 누적되는 특징이 있기 때문에 반드시 한번의 학습이 완료될 때마다 gradient를 0로 만들고 시작해야 한다.
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):  # train 모드일 때만  gradient 계산이 이뤄지도록 함

                    output = model(image, question)    
                    # 최대 확률을 갖는 값 선택
                    _, pred_exp1 = torch.max(output, 1) # exp1은 <unk>를 answer로 포함했을 경우
                    _, pred_exp2 = torch.max(output, 1) # exp2은 <unk>를 answer로 포함하지 않았을 경우
                    loss = criterion(output, label) # cross-entropy-loss 사용

                    if phase == 'train':
                        loss.backward() # gradient 계산
                        optimizer.step() # gradient descent를 이용한 최적화

                # <unk> : unknown token, 출연 빈도가 낮은 토큰은 모두 <unk>로 대체.
                # Evaluation metric of 'multiple choice'
                # Exp1: our model prediction to '<unk>' is accepted as the answer.
                # Exp2: our model prediction to '<unk>' is NOT accepted as the answer.

                pred_exp2[pred_exp2 == ans_unk_idx] = -9999 # <unk>를 answer로 사용하지 않을려고 아예 -9999를 넣어버림
                running_loss += loss.item() # 계산된 loss가 있을 때 loss의 scaler 값을 가져올 수 있음.
                running_corr_exp1 += torch.stack([(ans == pred_exp1.cpu()) for ans in multi_choice]).any(dim=0).sum()
                running_corr_exp2 += torch.stack([(ans == pred_exp2.cpu()) for ans in multi_choice]).any(dim=0).sum()

                if batch_idx % 100 == 0:
                    print('| {} SET | Epoch [{:02d}/{:02d}], Step [{:04d}/{:04d}], Loss: {:.4f}'
                          .format(phase.upper(), epoch+1, args.num_epochs, batch_idx, int(batch_step_size), loss.item()))

            epoch_loss = running_loss / batch_step_size
            epoch_acc_exp1 = running_corr_exp1.double() / len(data_loader[phase].dataset)      
            epoch_acc_exp2 = running_corr_exp2.double() / len(data_loader[phase].dataset)     

            print('| {} SET | Epoch [{:02d}/{:02d}], Loss: {:.4f}, Acc(Exp1): {:.4f}, Acc(Exp2): {:.4f} \n'
                  .format(phase.upper(), epoch+1, args.num_epochs, epoch_loss, epoch_acc_exp1, epoch_acc_exp2))

            # train log 기록
            with open(os.path.join(args.log_dir, '{}-log-epoch-{:02}.txt')
                      .format(phase, epoch+1), 'w') as f:
                f.write(str(epoch+1) + '\t'
                        + str(epoch_loss) + '\t'
                        + str(epoch_acc_exp1.item()) + '\t'
                        + str(epoch_acc_exp2.item()))
        
        # model save 
        if (epoch+1) % args.save_step == 0:
            torch.save({'epoch': epoch+1, 'state_dict': model.state_dict()},
                       os.path.join(args.model_dir, 'model-epoch-{:02d}.ckpt'.format(epoch+1)))



# 각종 parameters 값들을 불러오기 위한 argparse
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', type=str, default='./datasets',
                        help='input directory for visual question answering.')

    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='directory for logs.')

    parser.add_argument('--model_dir', type=str, default='./models',
                        help='directory for saved models.')

    parser.add_argument('--max_qst_length', type=int, default=30,
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

    parser.add_argument('--hidden_size', type=int, default=512,
                        help='hidden_size in the LSTM.')

    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate for training.')

    parser.add_argument('--step_size', type=int, default=10,
                        help='period of learning rate decay.')

    parser.add_argument('--gamma', type=float, default=0.1,
                        help='multiplicative factor of learning rate decay.')

    parser.add_argument('--num_epochs', type=int, default=30,
                        help='number of epochs.')

    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size.')

    parser.add_argument('--num_workers', type=int, default=8,
                        help='number of processes working on cpu.')

    parser.add_argument('--save_step', type=int, default=1,
                        help='save step of model.')

    args = parser.parse_args()

    main(args)
