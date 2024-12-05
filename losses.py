import torch
class Regressions:


    def mean_absolute_error(y_true, y_pred):
        return torch.mean(torch.abs(y_true - y_pred))

    def log_cosh_loss(y_true, y_pred):
        log_cosh = torch.log(torch.cosh(y_pred - y_true))
        return torch.mean(log_cosh)

    def mean_absolute_percentage_error(y_true, y_pred):
        epsilon = 1e-10
        percentage_error = torch.abs((y_true - y_pred) / torch.maximum(torch.abs(y_true), torch.tensor(epsilon, dtype=torch.float32)))
        return torch.mean(percentage_error) * 100
    def mean_square_error(y_true, y_pred):
        return torch.mean(torch.square(y_true - y_pred))
    def mean_squared_logarithmic_error(y_true, y_pred):
        epsilon = 1e-10
        log_error = torch.log((y_pred + 1) / (y_true + 1))
        return torch.mean(torch.square(log_error))

    def poisson_loss(y_true, y_pred):
        epsilon = 1e-10
        poisson_loss = y_pred - y_true * torch.log(torch.maximum(y_pred, epsilon))
        return torch.mean(poisson_loss)
    def relative_absolute_error(y_true, y_pred):

 
        abs_err = torch.abs(y_true - y_pred)
        mean_true = torch.mean(torch.abs(y_true - torch.mean(y_true)))
        rae = torch.mean(abs_err) / mean_true
        return rae
    def relative_squared_error(y_true, y_pred):
 
        sq_err = torch.square(y_true - y_pred)
        mean_true = torch.mean(torch.square(y_true - torch.mean(y_true)))
        rse = torch.mean(sq_err) / mean_true
        return rse
