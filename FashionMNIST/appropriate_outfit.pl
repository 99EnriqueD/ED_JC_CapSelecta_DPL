nn(fashion_mnist_net,[X], O, [0,1,2,3,4,5,6,7,8,9]) :: category(X,O).

% Appropriate Wardrobe ------------------------------------------------------------------------------------------------------------------

% Tshirt
rain(0,True).
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

% Wardrobe must include a coat or pullover.
goodForRain(Pieces,1) :- member(2,Pieces).
goodForRain(Pieces,1) :- member(4,Pieces).
goodForRain(Pieces,0) :- \+ (member(2,Pieces) ; member(4,Pieces))

% Wardrobe must not include sandals nor a tshirt.
formal(Pieces,0) :- member(5,Pieces).
formal(Pieces,0) :- member(3,Pieces).
formal(Pieces,1) :- \+ (member(5,Pieces) ; member(3,Pieces))

% Wardrobe must include sandals and a tshirt.
goodForWarm(Pieces,1) :- member(0,Pieces), member(5,Pieces).
goodForWarm(Pieces,0) :- \+ (member(0,Pieces), member(5,Pieces)).


% Whether the pieces int the given wardrobe make a full outfit.
fullOutfit(Pieces,1) :- hasShoe(Pieces), hasTop(Pieces), hasBottoms(Pieces).
fullOutfit(Pieces,1) :- hasShoe(Pieces), hasDress(Pieces), hasBag(Pieces).
fullOutfit(Pieces,1) :- hasShoe(Pieces), hasDress(Pieces), hasCoat(Pieces).
fullOutfit(Pieces,0).

hasShoe(Pieces) :- member(9,Pieces) ; member(7,Pieces) ; member(5,Pieces).
hasTop(Pieces) :- member(0,Pieces) ; member(2,Pieces) ; member(6,Pieces) ; member(4,Pieces).
hasBottoms(Pieces) :- member(1,Pieces).
hasDress(Pieces) :- member(3,Pieces).
hasBag(Pieces) :- member(8,Pieces).
hasCoat(Pieces) :- member(4,Pieces).


appropriateWardrobe(I1, I2, I3, Rain, Formal, Warm, Full) :- 
    category(I1,C1),
    category(I2,C2),
    category(I3,C3),
    Pieces = [C1,C2,C3],
    goodForRain(Pieces,Rain),
    goodForWarm(Pieces,Warm),
    formal(Pieces,Formal),
    fullOutfit(Pieces,Full). % TODO: make sure backtracking doesn't mess things up!
    





    
