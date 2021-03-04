import Model
import data
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


if __name__ == '__main__':

    train_loader, test_loader = data.get_data()

    model = Model.CoolModel()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    if torch.cuda.is_available():
        criterion.cuda()
        model.cuda()

    epochs = 30

    train_losses = []
    test_losses = []
    train_correct = []
    test_correct = []

    for epoch in range(epochs):
        trn_corr = 0
        tst_corr = 0
        model.train()

        for b, (X_train, y_train) in enumerate(train_loader):
            b+=1

            if torch.cuda.is_available():
                X_train = X_train.cuda()
                y_train = y_train.cuda()

            y_pred = model.forward(X_train)

            optimizer.zero_grad()
            loss = criterion(y_pred, y_train)

            loss.backward()
            optimizer.step()


            predicted = torch.max(y_pred.data, 1)[1]
            batch_corr = (predicted == y_train).sum()
            trn_corr += batch_corr

        print(f'epoch: {epoch}  loss: {loss.item()}   accuracy: {trn_corr.item()*100/1895}')
        train_losses.append(loss)
        train_correct.append(trn_corr)
        model.eval()

        with torch.no_grad():
            tst_cor = 0
            for X_test,y_test in test_loader:
                if torch.cuda.is_available():
                    X_test = X_test.cuda()
                    y_test = y_test.cuda()

                y_pred = model.forward(X_test)

                predicted = torch.max(y_pred.data, 1)[1]
                batch_corr = (predicted == y_test).sum()
                tst_corr += batch_corr

        test_correct.append(tst_corr)
        test_losses.append(loss)
    
    print(f'test accuracy: {test_correct[len(test_correct)-1]*100/500}%')


plt.plot([t/20 for t in train_correct], label='training accuracy')
plt.plot([t/5 for t in test_correct], label='validation accuracy')
plt.title('Accuracy at the end of each epoch')
plt.legend()
plt.show()


