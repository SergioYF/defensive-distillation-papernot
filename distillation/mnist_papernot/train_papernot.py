from utils import *

def generate_sigmas():
    """
    Generate uncertainty array using pretrained mnist model and store it as file.
    :return: the uncertainty array for training set, shape=(60000,)
    """
    model = Net()
    if args.cuda:
        model.cuda()
    state_dict = load_dict()
    model.load_state_dict(state_dict)
    start = time()
    uncertainty = relabel(model, normalize=True)
    end = time()
    print("Relabeling costs {}s".format(end - start))
    print("Uncertainty: ")
    print(uncertainty)
    return uncertainty

def generate_papernot():
    """
    Train the second/distilled model using uncertaity array
    :return: distilled model
    """
    sigmas = np.load(PROJ_PATH + 'mnist_papernot/sigmas.npz')['sigmas']
    start = time()
    model_papernot = train_papernot(sigmas)
    end = time()
    print("Papernot training costs {}s".format(end - start))
    return model_papernot

def test_papernot_on_test_set(model_papernot=None):
    """
    Test performance of distilled model using mnist test set
    """
    if model_papernot is None:
        model_papernot = Net(output_units=11)
        if args.cuda:
            model_papernot.cuda()
        state_dict = load_dict(pth_path=PROJ_PATH + 'mnist_papernot/mnist_papernot.pth')
        model_papernot.load_state_dict(state_dict)
    sigmas = np.load(PROJ_PATH + 'mnist_papernot/test_set_sigmas.npy')
    model_papernot.eval()
    loader = get_test_loader()
    test_papernot_model(model_papernot, sigmas, loader)
    inspect_model_result()

def test_papernot_on_training_set(model_papernot=None):
    """
    Test performance of distilled model using mnist training set
    :return:
    """
    if model_papernot is None:
        model_papernot = Net(output_units=11)
        if args.cuda:
            model_papernot.cuda()
        state_dict = load_dict(pth_path=PROJ_PATH + 'mnist_papernot/mnist_papernot.pth')
        model_papernot.load_state_dict(state_dict)
    sigmas = np.load(PROJ_PATH + 'mnist_papernot/sigmas.npz')['sigmas']
    model_papernot.eval()
    loader = get_train_loader()
    test_papernot_model(model_papernot, sigmas, loader)
    inspect_model_result(PROJ_PATH + 'mnist_papernot/result_{}.npy'.format(args.alpha))

def label_test_set():
    model = Net()
    state_dict = load_dict('./mnist_train.pth')
    model.load_state_dict(state_dict)
    test_unc = relabel(model=model, data_loader=get_test_loader(), normalize=False)
    file = np.load('sigmas.npz')
    base = file['base']
    final_unc = test_unc / base
    final_unc = final_unc.clip(max=1)
    np.save('test_set_sigmas', final_unc)

if __name__ == '__main__':



    def run():
        # these 5 operation can be executed respectively under the given order
        # sigmas = generate_sigmas()
        # model_papernot = generate_papernot()
        # label_test_set()
        test_papernot_on_training_set()
        test_papernot_on_test_set()

    def multiple_uncertainty_test(repeat_times=20, dropout_times=20, generate=False):
        # It's just an additional test and could be ignored
        sigmas = []
        max_list = []
        if generate:
            model = Net()
            if args.cuda:
                model.cuda()
            state_dict = load_dict()
            model.load_state_dict(state_dict)
            for i in range(repeat_times):
                uncertainty = relabel(model, test_times=dropout_times, test_size=1000, save_mark=i, normalize=False)
                sigma = process_sigma(uncertainty, save_mark=i)
                sigmas.append(sigma)
                max_list.append(uncertainty.argmax())
        else:
            str = './sigma_{}.npy'
            for i in range(repeat_times):
                s = np.load(str.format(i))
                sigmas.append(s)
                max_list.append(s.argmax())
        max_inspect = np.zeros((repeat_times, repeat_times))
        mean_diff = np.zeros((repeat_times, repeat_times))
        for i in range(repeat_times):
            for j in range(repeat_times):
                max_inspect[i, j] = sigmas[i][max_list[j]]
                mean_diff[i, j] = np.mean(np.abs(sigmas[i]-sigmas[j]))
        print(max_inspect)
        print(max_inspect.mean())
        print(mean_diff)
        print(mean_diff.mean())

    def test_uncertainty():
        # It's just an additional test and could be ignored
        uncs = []
        base = []
        maxs = []
        argmaxs = []
        sigmas = []
        for i in range(20):
            unc = np.load('uncertainty_{}.npy'.format(i))
            uncs.append(unc)
            m = unc.mean()
            s = np.sqrt(np.mean(np.square(unc - m)))
            base_i = m + 3 * s
            base.append(base_i)
            argmaxs.append(unc.argmax())
            maxs.append(unc.max())
            sigmas.append(unc / base_i)
        max_inspect = np.zeros((20, 20))
        mean_diff = np.zeros((20, 20))
        for i in range(20):
            for j in range(20):
                max_inspect[i, j] = sigmas[i][argmaxs[j]]
                mean_diff[i, j] = np.mean(np.abs(sigmas[i]-sigmas[j]))
        print(max_inspect)
        print(max_inspect.mean())
        print(mean_diff)
        print(mean_diff.mean())

        length = sigmas[0].shape[0]
        sigmas = np.array(sigmas)
        error1 = np.zeros(length)
        error2 = np.zeros(length)
        for j in range(length):
            m = sigmas[:, j].mean()
            s1 = np.mean(np.abs(sigmas[:, j] - m))
            s2 = np.sqrt(np.mean(np.square(sigmas[:, j] - m)))
            error1[j] = s1
            error2[j] = s2

        embed()

    run()