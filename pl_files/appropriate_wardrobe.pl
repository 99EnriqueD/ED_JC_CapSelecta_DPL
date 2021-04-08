nn(fashion_mnist_net,[X], O, [0,1,2,3,4,5,6,7,8,9]) :: category(X,O).

%Tshirt = 0
%Trouser = 1
%Pullover = 2
%Dress = 3
%Coat = 4
%Sandal = 5
%Shirt = 6
%Sneaker = 7
%Bag = 8
%Ankleboot = 9

%%%%%%%% Problog non-built ins
member(El, [H|T]) :- member_(T, El, H).
member_(_, El, El).
member_([H|T], El,_) :- member_(T, El, H).

% once(Goal) :- call(Goal),!.

memberchk(El, L) :- once(member(El,L)).
%%%%%%%%

% Appropriate Wardrobe ------------------------------------------------------------------------------------------------------------------

% Wardrobe must include a coat or pullover.
goodForRain(Pieces,1) :- memberchk(2,Pieces).
goodForRain(Pieces,1) :- memberchk(4,Pieces).
goodForRain(Pieces,0).
% goodForRain(Pieces,0) :- \+ (memberchk(2,Pieces) ; memberchk(4,Pieces)).

% Wardrobe must not include sneakers.
formal(Pieces,0) :- memberchk(7,Pieces).
formal(Pieces,1).
% formal(Pieces,1) :- \+ (memberchk(5,Pieces) ; memberchk(3,Pieces)).

% Wardrobe must include sandals or a tshirt.
goodForWarm(Pieces,1) :- memberchk(0,Pieces).
goodForWarm(Pieces,1) :- memberchk(5,Pieces).
goodForWarm(Pieces,0).
% goodForWarm(Pieces,0) :- \+ (memberchk(0,Pieces), memberchk(5,Pieces)).


% Whether the pieces int the given wardrobe make a full outfit.
fullOutfit(Pieces,1) :- hasShoe(Pieces), hasTop(Pieces), hasBottoms(Pieces).
fullOutfit(Pieces,1) :- hasShoe(Pieces), hasDress(Pieces), hasBag(Pieces).
fullOutfit(Pieces,1) :- hasShoe(Pieces), hasDress(Pieces), hasCoat(Pieces).
fullOutfit(Pieces,0).

hasShoe(Pieces) :- memberchk(9,Pieces).
hasShoe(Pieces) :- memberchk(7,Pieces). 
hasShoe(Pieces) :- memberchk(5,Pieces).
hasTop(Pieces) :- memberchk(0,Pieces). 
hasTop(Pieces) :- memberchk(2,Pieces).
hasTop(Pieces) :- memberchk(6,Pieces).
hasTop(Pieces) :-  memberchk(4,Pieces).
hasBottoms(Pieces) :- memberchk(1,Pieces).
hasDress(Pieces) :- memberchk(3,Pieces).
hasBag(Pieces) :- memberchk(8,Pieces).
hasCoat(Pieces) :- memberchk(4,Pieces).


appropriateWardrobe(I1, I2, I3, Rain, Formal, Warm, Full) :- 
    category(I1,C1),
    category(I2,C2),
    category(I3,C3),
    goodForRain([C1,C2,C3],Rain),
    goodForWarm([C1,C2,C3],Warm),
    formal([C1,C2,C3],Formal),
    fullOutfit([C1,C2,C3],Full).
    % NOTE: make sure backtracking doesn't mess things up!
    





    
