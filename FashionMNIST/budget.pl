nn(fashion_mnist_net,[X], O, [0,1,2,3,4,5,6,7,8,9]) :: category(X,O).

% BUDGET ------------------------------------------------------------------------------------------------------------------

% piecePrice(Piece,PiecePrice) :- True. % TODO, could be based on similarity to clothes of known price or could give price distributions of  

% outfitPrice([Piece|PS],OutfitPrice) :- 
%     outfitPrice(Ps,RestPrice), 
%     OutfitPrice is RestPrice + piecePrice(Piece).

% withinBudget(Pieces,Budget) :- outfitPrice(Pieces,OutfitPrice), Budget >= OutfitPrice.

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

totalPrice(I1, I2, TP) :- 
    category(I1,C1),
    category(I2,C2),
    price(C1,P1),
    price(C2,P2),
    TP is P1 + P2.
