import pandas as pd, matplotlib.pyplot as plt, catboost as catb, statsmodels as sm, statsmodels.tsa as tsa, statsmodels.graphics.tsaplots, statsmodels.tsa.arima.model, statsmodels.tsa.stattools, itertools as it, warnings, pmdarima.arima as pm
from sklearn.model_selection import train_test_split
from sys import argv
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from time import time
from concurrent.futures import ProcessPoolExecutor

warnings.simplefilter("ignore", ConvergenceWarning)
warnings.simplefilter("ignore", UserWarning)


def ARIMA_make_fore(target, eval, n_lags=10, n_pred=10):
    print("Starting ARIMA subroutine...")
    startTime = time()
    diffed = target.copy()
    d = 0
    while tsa.stattools.adfuller(diffed.dropna())[1] > 0.05:
        diffed = diffed.diff().dropna()
        d += 1
    acf, acf_conf = statsmodels.tsa.stattools.acf(
        diffed.dropna(), nlags=n_lags, alpha=0.05, fft=True
    )
    pacf, pacf_conf = statsmodels.tsa.stattools.pacf(
        diffed.dropna(), nlags=n_lags, alpha=0.05
    )
    mas = [0]
    ars = [0]
    for i in range(1, len(acf)):
        if 2 * acf[i] < acf_conf[i][0] or 2 * acf[i] > acf_conf[i][1]:
            mas.append(i)
    for i in range(1, len(pacf)):
        if 2 * pacf[i] < pacf_conf[i][0] or 2 * pacf[i] > pacf_conf[i][1]:
            ars.append(i)
    models = dict()
    results = dict()
    for p, q in it.product(ars, mas):
        models[(p, q)] = tsa.arima.model.ARIMA(target, order=(p, d, q))
        results[(p, q)] = models[(p, q)].fit()
    pars = min(results.items(), key=lambda x: x[1].aicc)[0]
    result = results[pars]
    preds = pd.Series(result.forecast(eval.size), index=eval.index)
    confs = pd.DataFrame(result.get_forecast(eval.size).conf_int(), index=eval.index)
    preds.rename("predicted data", inplace=True)
    eval.plot(legend=True)
    preds.plot(legend=True)
    plt.fill_between(
        confs.index,
        confs.iloc[:, 0],
        confs.iloc[:, 1],
        alpha=0.3,
        label="95% confidence intervals",
    )
    plt.legend()
    plt.ticklabel_format(useOffset=False, style="plain")
    plt.savefig("ARIMA.log.png")
    plt.close()
    RMSE = ((eval - preds) ** 2).mean() ** 0.5
    with open("ARIMA.log", "w") as log:
        print(result.summary(), file=log)
        print("\n\n\nRMSE on evaluation set: ", RMSE, file=log)
    result.append(eval.values, refit=True)
    fore = result.forecast(n_pred)
    conf_fore = result.get_forecast(n_pred).conf_int()
    print("ARIMA done in", int(time() - startTime), "seconds with RMSE:", RMSE)
    return RMSE, fore.values, conf_fore.values


def AutoARIMA_make_fore(target, eval, n_pred=10):
    print("Starting auto ARIMA subroutine...")
    startTime = time()
    model = pm.auto_arima(target)
    preds, confs = model.predict(eval.size, return_conf_int=True)
    preds = pd.Series(preds, index=eval.index)
    confs = pd.DataFrame(confs, index=eval.index)
    preds.rename("predicted data", inplace=True)
    eval.plot(legend=True)
    preds.plot(legend=True)
    plt.fill_between(
        confs.index,
        confs.iloc[:, 0],
        confs.iloc[:, 1],
        alpha=0.3,
        label="95% confidence intervals",
    )
    plt.legend()
    plt.ticklabel_format(useOffset=False, style="plain")
    plt.savefig("auto_ARIMA.log.png")
    plt.close()
    RMSE = ((eval - preds) ** 2).mean() ** 0.5
    with open("auto_ARIMA.log", "w") as log:
        print(model.summary(), file=log)
        print("\n\n\nRMSE on evaluation set: ", RMSE, file=log)
    model.update(eval, refit=True)
    fore, conf_fore = model.predict(n_pred, return_conf_int=True)
    print("auto ARIMA done in", int(time() - startTime), "seconds with RMSE:", RMSE)
    return RMSE, fore, conf_fore


def Boosting_make_fore(target, eval, n_lags=100, n_pred=10):
    print("Starting boosting subroutine...")
    startTime = time()
    eval_start = target.iat[-1]
    dtarget = target.diff().dropna()
    dev = pd.DataFrame({"LAG 1": dtarget.shift(1)})
    for i in range(1, n_lags):
        dev["LAG " + str(1 + i)] = dev["LAG " + str(i)].shift(1)
    dev = dev[n_lags:]
    dtarget = dtarget[n_lags:]
    x_train, x_test, y_train, y_test = train_test_split(
        dev, dtarget, test_size=0.2, shuffle=False
    )
    train = catb.Pool(x_train, y_train)
    test = catb.Pool(x_test, y_test)
    try:
        model = catb.CatBoostRegressor(
            iterations=100000,
            task_type="GPU",
            use_best_model=True,
            early_stopping_rounds=1000,
        )
        model.fit(train, eval_set=test)
    except catb.CatBoostError:
        model = catb.CatBoostRegressor(
            iterations=100000,
            task_type="CPU",
            use_best_model=True,
            early_stopping_rounds=1000,
        )
        model.fit(train, eval_set=test, verbose=False)
    cur = x_test[-1:].copy()
    res = [eval_start]
    for _ in range(eval.size):
        new = model.predict(cur)[0]
        res.append(res[-1] + new)
        cur = cur.shift(1, axis=1, fill_value=new)
    preds = pd.Series(res[1:], index=eval.index)
    preds.rename("predicted data", inplace=True)
    eval.plot(legend=True)
    preds.plot(legend=True)
    plt.ticklabel_format(useOffset=False, style="plain")
    plt.savefig("boosting.log.png")
    plt.close()
    RMSE = ((eval - preds) ** 2).mean() ** 0.5
    fore_start = eval.iat[-1]
    cur = pd.concat([target, eval]).diff().dropna()
    fore_features = dict()
    for i in range(1, 101):
        fore_features["LAG " + str(i)] = cur.iloc[-i]
    fore_features = pd.DataFrame(fore_features, index=[0])
    rtrn = [fore_start]
    for _ in range(n_pred):
        new = model.predict(fore_features)[0]
        rtrn.append(rtrn[-1] + new)
        fore_features = fore_features.shift(1, axis=1, fill_value=new)
    print("boosting done in", int(time() - startTime), "seconds with RMSE:", RMSE)
    return RMSE, rtrn[1:]


if __name__ == "__main__":
    data = argv[1]
    n_pred = 10
    if len(argv) > 2:
        n_pred = int(argv[2])
    ts = pd.read_csv(data, index_col=False, squeeze=True)
    target, eval = train_test_split(ts, test_size=0.01, shuffle=False)
    eval.rename("evaluation data", inplace=True)
    with ProcessPoolExecutor() as executor:
        autoARIMA_proc = executor.submit(
            AutoARIMA_make_fore, target, eval, n_pred=n_pred
        )
        ARIMA_proc = executor.submit(ARIMA_make_fore, target, eval, n_pred=n_pred)
        boosting_proc = executor.submit(Boosting_make_fore, target, eval, n_pred=n_pred)
        autoARIMA_RMSE, autoARIMA_fore, autoARIMA_conf_fore = autoARIMA_proc.result()
        ARIMA_RMSE, ARIMA_fore, ARIMA_conf_fore = ARIMA_proc.result()
        boosting_RMSE, boosting_fore = boosting_proc.result()
    if boosting_RMSE < autoARIMA_RMSE and boosting_RMSE < ARIMA_RMSE:
        print("Using boosting result...")
        with open("result.csv", "w") as res:
            print("forecast", file=res)
            print(*boosting_fore, sep="\n", end="", file=res)
        print("Saved forecast to result.csv")
        plt.plot(boosting_fore, label="forecast")
        plt.legend()
        plt.ticklabel_format(useOffset=False, style="plain")
        plt.savefig("result.png")
        plt.close()
        print("Saved forecast plot to result.png")
    elif autoARIMA_RMSE < ARIMA_RMSE:
        print("Using auto ARIMA result...")
        with open("result.csv", "w") as res:
            print("forecast,conf int low,conf int high", file=res)
            for i in range(len(autoARIMA_fore)):
                print(
                    autoARIMA_fore[i],
                    autoARIMA_conf_fore[i][0],
                    autoARIMA_conf_fore[i][1],
                    sep=",",
                    file=res,
                )
        print("Saved forecast to result.csv")
        plt.plot(autoARIMA_fore, label="forecast")
        plt.fill_between(
            range(len(autoARIMA_conf_fore)),
            [i[0] for i in autoARIMA_conf_fore],
            [i[1] for i in autoARIMA_conf_fore],
            alpha=0.3,
            label="95% confidence intervals",
        )
        plt.legend()
        plt.ticklabel_format(useOffset=False, style="plain")
        plt.savefig("result.png")
        plt.close()
        print("Saved forecast plot to result.png")
    else:
        print("Using ARIMA result...")
        with open("result.csv", "w") as res:
            print("forecast,conf int low,conf int high", file=res)
            for i in range(len(ARIMA_fore)):
                print(
                    ARIMA_fore[i],
                    ARIMA_conf_fore[i][0],
                    ARIMA_conf_fore[i][1],
                    sep=",",
                    file=res,
                )
        print("Saved forecast to result.csv")
        plt.plot(ARIMA_fore, label="forecast")
        plt.fill_between(
            range(len(ARIMA_conf_fore)),
            [i[0] for i in ARIMA_conf_fore],
            [i[1] for i in ARIMA_conf_fore],
            alpha=0.3,
            label="95% confidence intervals",
        )
        plt.legend()
        plt.ticklabel_format(useOffset=False, style="plain")
        plt.savefig("result.png")
        plt.close()
        print("Saved forecast plot to result.png")
