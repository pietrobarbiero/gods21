

def main():
    x, y = make_classification(n_samples=1000, n_features=10, n_informative=2, n_classes=2,
                               random_state=42, )
    # x, y = load_iris(return_X_y=True)
    # scaler = StandardScaler()
    # x = scaler.fit_transform(x)

    layers = [
        te.nn.EntropyLinear(x.shape[1], 100, n_classes=len(np.unique(y))),
        # torch.nn.Linear(x.shape[1], 100),
        torch.nn.LeakyReLU(),
        # torch.nn.Linear(30, 10),
        # torch.nn.LeakyReLU(),
        torch.nn.Linear(100, 1),
        # torch.nn.Linear(100, len(np.unique(y))),
    ]
    model = torch.nn.Sequential(*layers)
    optimizer = 'adamw'  #
    optim = torch.optim.AdamW(model.parameters(), lr=0.001)
    # loss_form = torch.nn.BCEWithLogitsLoss()
    loss_form = torch.nn.CrossEntropyLoss()
    # for layer in model.children():
    #     if hasattr(layer, 'reset_parameters'):
    #         layer.reset_parameters()
    # model.train()
    # for epoch in range(10000):
    #     optim.zero_grad()
    #     y_pred = model(torch.FloatTensor(x)).squeeze(-1)
    #     loss = loss_form(y_pred, torch.LongTensor(y))
    #     loss.backward()
    #     optim.step()
    #     if epoch % 100 == 0:
    #         print(f'Epoch {epoch}/{10000}: {loss.item()}')

    evo_lens = EvoLENs(model, optimizer, loss_form, compression='features', train_epochs=10,
                       max_generations=100, lr=0.001,)
    evo_lens.fit(x, y)

    result_dir = './evolens_results/'
    os.makedirs(result_dir, exist_ok=True)
    joblib.dump(evo_lens, f'{result_dir}evo_lens.pkl')
    evo_lens2 = joblib.load(f'{result_dir}evo_lens.pkl')
    joblib.dump(evo_lens.best_set_, f'{result_dir}evo_lens_best.joblib')
    joblib.dump(evo_lens.optimal_solutions_, f'{result_dir}evo_lens_solutions.joblib')
    joblib.dump(evo_lens.feature_ranking_, f'{result_dir}evo_lens_feature_ranking.joblib')

    print()
    print('Best individual\'s features:')
    print(evo_lens.best_set_['features'])
    print()
    print('Best individual\'s F1:')
    print(evo_lens.best_set_['accuracy'])
    print()
    print('Best individual\'s explanation accuracy:')
    print(evo_lens.best_set_['explanation_accuracy'])
    print()
    print('Best individual\'s explanation complexity:')
    print(evo_lens.best_set_['explanation_complexity'])
    print()


if __name__ == '__main__':
    main()