def ImagePipeline(inp, show = False, shape = (32, 32)) : 
    img = Image.open(inp)
    img = img.resize(shape)
    img = np.array(img)
    if(show) : 
        plt.imshow(img, cmap = "gray")
        plt.show()
    img = img.flatten()
    img = torch.from_numpy(img)
    img = img.type(torch.int64)
    return img