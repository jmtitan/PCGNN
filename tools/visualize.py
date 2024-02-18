import matplotlib.pyplot as plt

'''
绘制常用内部tensor对比图
'''
def plot(show_dogs, show_cats):
    plt.figure(figsize=(10, 2))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for idx, inp in enumerate(show_dogs):
        inp_origin = inp[0].numpy().transpose((1, 2, 0))
        inp_noise = inp[1].numpy().transpose((1, 2, 0))
        plt.subplot(2,10,idx+1)
        plt.imshow(inp_origin)
        plt.xticks(())
        plt.yticks(())
        plt.title("dog{}".format(idx))
        plt.subplot(2,10,idx+1+10)
        plt.imshow(inp_noise)
        plt.xticks(())
        plt.yticks(())

    plt.figure(figsize=(10, 2))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for idx, inp in enumerate(show_cats):
        inp_origin = inp[0].numpy().transpose((1, 2, 0))
        inp_noise = inp[1].numpy().transpose((1, 2, 0))
        plt.subplot(2,10,idx+1)
        plt.imshow(inp_origin)
        plt.xticks(())
        plt.yticks(())
        plt.title("cat{}".format(idx))
        plt.subplot(2,10,idx+1+10)
        plt.imshow(inp_noise)
        plt.xticks(())
        plt.yticks(())