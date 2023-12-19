
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from Scripts import reverse_preprocessing_func as rv
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
from Scripts import to_stationary as ts
from Scripts import dir_functions
from Scripts import metrics_plotting as mp
import json
import copy

torch.manual_seed(32)


def training_and_tuning(model_class, data, epochs, min_epoch_eval, df_path_name, params_grid, plot_results=True):
    dir_functions.create_validation_folder(df_path_name)
    fig, ax = plt.subplots(figsize=(24, 8))

    # Try different sets of parameters from params_grid
    for i,g in enumerate(params_grid):
        n1 = g['module__n1']
        lr = g['lr']

        # Set model, criterion and optimizer to train
        model = model_class(n1)
        criterion=nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Set val path and prepare folder
        lr_save = str(lr).split('.')[1] # learning rate save name
        model_val_path = df_path_name + f'/validation/val_{n1}_{lr_save}'

        dir_functions.check_model_path(model_val_path)

        # train
        best_error= np.inf
        best_mape=0
        best_epoch=0
        train_mse=[]
    

        for epoch in range(epochs):
            batch_mse = []
            model.train(True)
            epoch+=1
            
            for batch_index, batch in enumerate(data['train_loader']):
                x_batch, y_batch = batch[0], batch[1]
                
            
                output = model(x_batch)
                loss = criterion(output, y_batch)
            
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                batch_mse.append(loss.detach().numpy())
                
            train_mse.append(np.mean(batch_mse))

            if (epoch >= min_epoch_eval): # Usually the last 20-50 epochs
                model.train(False)
                
                with torch.no_grad():
                    y_val_pred = model(data['x_val'])
                    df_val_pred = mp.pred_to_dataframe(y_val_pred, data['y_val_date'])
                    mse, mape = mp.calc_metrics(data['df_val_norm'], df_val_pred)['mse'], mp.calc_metrics(data['df_val_norm'], df_val_pred)['mape']
                
                    batch_val_mse = []
                    
                    for batch_index, batch in enumerate(data['val_loader']):
                        x_batch, y_batch = batch[0], batch[1]
                        
                        output = model(x_batch) # predict
                        loss = criterion(output, y_batch)
                        batch_val_mse.append(loss.detach().numpy())
                    val_mse = np.mean(batch_val_mse)
                
                          
                # Save only if error is less than before
                if mse <= best_error: # Save only if error is less than before
                    best_error = mse
                    best_epoch = epoch
                    torch.save({
                        'epoch': epochs,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': criterion,
                        }, f'{model_val_path}/model_{str(epoch)}.pth')


        if plot_results:
              plt.subplot(2, int(len(params_grid)/2), i+1)
              plt.plot(train_mse)
              plt.title(f'lr={lr}, n1={n1}')
              print(f'lr = {lr}, n1 = {n1}, min val mse epoch = {best_epoch}, min val MSE = {best_error}, min val MAPE = {best_mape}')







def evaluation(model, data, epochs, optimizer, criterion, model_path, min_epoch_eval, denorm, denorm_kwargs={}):
    train_mse = []
    best_error= np.inf
    best_model = copy.deepcopy(model)
    best_epoch=0
    
    dir_functions.check_model_path(model_path)

    # Fit data in train + val sets, and evaluate on test set
    for epoch in range(epochs):
        batch_mse =[]
        epoch+=1
        model.train(True)
        
        for batch_index, batch in enumerate(data['train_loader']):
            x_batch, y_batch = batch[0], batch[1]
            
            
            output = model(x_batch)
            loss = criterion(output, y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            batch_mse.append(loss.detach().numpy())
            
        train_mse.append(np.mean(batch_mse))
        

        if (epoch >= min_epoch_eval): # Usually the last 20-50 epochs
            # Eval epoch based on mse between predicted and test set, both denormalized
            model.train(False)
            
            with torch.no_grad():
                y_val_pred = model.predict(data['x_val'])
                df_val_pred = mp.pred_to_dataframe(y_val_pred, data['y_val_date'])
                mse, mape = mp.calc_metrics(data['df_val_norm'], df_val_pred)['mse'], mp.calc_metrics(data['df_val_norm'], df_val_pred)['mape']

            
                batch_val_mse = []
                for batch_index, batch in enumerate(data['val_loader']):
                    x_batch, y_batch = batch[0], batch[1]
                    
                    output = model(x_batch) # predict
                    loss = criterion(output, y_batch)
                    batch_val_mse.append(loss.detach().numpy())
                val_mse = np.mean(batch_val_mse)
            
            # Save only if error is less than before
            if mse <= best_error: # Save only if error is less than before
                best_model = copy.deepcopy(model)
                torch.save({
                    'epoch': epochs,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': criterion,
                    }, f'{model_path}/model_{str(epoch)}.pth')

    y_test_pred = best_model.predict(data['x_test'])
    df_test_pred = denorm(data['df_test'], y_test_pred, data['y_test_date'], denorm_kwargs)
    test_mse = mp.calc_metrics(data['df_test'], df_test_pred)['mse']
    test_mape = mp.calc_metrics(data['df_test'], df_test_pred)['mape']

    results = {
        'best_model': best_model,
        'test_mse': test_mse,
        'test_mape': test_mape,
        'df_denorm_test_pred' : df_test_pred
    }

    return results




def multiple_eval(model_class, model_n1, model_lr, data, epochs, df_path_name, n_eval, denorm, min_epoch_eval, verbose=True, denorm_kwargs={}):

    '''
    df_path_name = name of the df path to save the training models
    n_train = numbers of trains
    predict_and_denorm = function that makes a batch of predictions and denormalize the result
    data =  dictionary of train and test data: {'df_test': df_test, 'x_train': x_train, 'x_test': x_test, 'y_train': y_train, 'y_train_date': y_train_date, 'y_test_date': y_test_date}
    '''
    mse_list = []
    mape_list = []

    for i in range(n_eval):
        dir_functions.erase_train_files(df_path_name)
        best_model = model_class(model_n1)
        criterion=nn.MSELoss()
        optimizer = torch.optim.Adam(best_model.parameters(), lr=model_lr)

        results = evaluation(model=best_model, data=data,
                            epochs=epochs,
                            optimizer=optimizer, criterion=criterion,
                            model_path=df_path_name,
                            min_epoch_eval=min_epoch_eval,
                            denorm=denorm,
                            denorm_kwargs=denorm_kwargs)
        mse_list.append(results['test_mse'])
        mape_list.append(results['test_mape'])

        if verbose:
            mape = results['test_mape']
            print(f'Treino = {i}, Test MAPE = {mape}')

    metrics_list = {'mse_list': mse_list, 'mape_list': mape_list}
    # returns last best model
    return best_model, metrics_list, results['df_denorm_test_pred']