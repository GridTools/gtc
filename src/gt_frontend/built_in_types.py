class BuiltInTypeMeta(type):
    def __new__(cls, class_name, bases, namespace, args=None):
        assert bases == () or (len(bases)==1 and issubclass(bases[0], BuiltInType))
        assert all(attr[0:2] == "__" for attr in namespace.keys()) # no custom attributes
        # tülülü
        instance = type.__new__(cls, class_name, bases, namespace)
        instance.class_name = class_name
        instance.namespace = namespace
        instance.args = args
        return instance

    def __getitem__(self, args): # todo: evaluate __class_getitem__
        if not isinstance(args, tuple):
            args = (args,)
        return BuiltInTypeMeta(self.class_name, (), self.namespace, args=args)

    def __instancecheck__(self, instance):
        raise RuntimeError()

    def __subclasscheck__(self, other):
        if self.namespace == other.namespace and self.class_name == other.class_name:
            if self.args == None or self.args == other.args:
                return True
        return False

class BuiltInType(metaclass=BuiltInTypeMeta):
    pass

class Mesh(BuiltInType):
    pass

class Field(BuiltInType):
    pass

class TemporaryField(BuiltInType):
    pass

class Location(BuiltInType):
    pass
