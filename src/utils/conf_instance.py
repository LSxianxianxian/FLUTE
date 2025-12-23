import inspect


class Instance:
    """
    class Demo():
        def __init__(self, x):
            self.x = x

        def get(self):
            return x

    class DemoInst(Instance):
        def __init__(self):
            super().__init__(
                target = demo.Demo,
                x = 2,
            )

    instance = conf.DemoInst.instantiate()
    """

    def __init__(self, target, **kwargs):
        if type(target) is list:
            self.target = []
            self.kwargs = []
            for item, d in target:
                self.target.append(item)
                self.kwargs.append(kwargs | d)
        else:
            self.target = target
            self.kwargs = kwargs

    @classmethod
    def instantiate(cls):
        instance = cls()
        if type(instance.target) is list:
            return [target(**kwargs) for target, kwargs in zip(instance.target, instance.kwargs)]
        else:
            kwargs = {k: v if not inspect.isclass(v) or not issubclass(v, Instance) else v.instantiate() for k, v in instance.kwargs.items()}
            return instance.target(**kwargs)