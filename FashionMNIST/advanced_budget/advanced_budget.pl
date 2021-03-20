nn(fashion_df_net,[X], O, [0,1,2,3,4,5,6]) :: category(X,O).

%shirt achtig 0 
%trui achtig 1
%short 2
%broek casual 3
%broek normaal 4
%skirt + dress 5
%coat 6

% Shirt (tank vs general vs button down)
t(0.25) :: price(0,5); t(0.5) :: price(0,10); t(0.25) :: price(0,15).
% Pullover
price(1,25).
% Shorts
price(2,20).
% Casual pants
price(3,25).
% Normal pants (general vs chinos)
t(0.666) :: price(4,40); t(0.333) :: price(4,45).
% Dress, skirt, etc
price(5,35).
% Coat (In general vs anarok)
t(0.666) :: price(6,40); t(0.333) :: price(6,50).


totalPrice(I1, I2, TP) :- 
    category(I1,C1),
    category(I2,C2),
    price(C1,P1),
    price(C2,P2),
    TP is P1 + P2.
