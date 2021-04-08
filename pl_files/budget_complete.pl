nn(fashion_mnist_net,[X], O, [0,1,2,3,4,5,6,7,8,9]) :: category(X,O).

% BUDGET ------------------------------------------------------------------------------------------------------------------

% Tshirt
price(0,10).
%Trouser
price(1,25).
%Pullover
price(2,20).
%Dress
price(3,25).
%Coat
price(4,50).
%Sandal
price(5,40).
%Shirt
price(6,20).
%Sneaker
price(7,30).
%Bag
price(8,60).
%Ankleboot
price(9,20).


totalPrice([],0).
totalPrice([I|Is], TP) :- 
    category(I,C),
    price(C,P),
    totalPrice(Is,PP),
    TP is P + PP.
