nn(advanced_budget_net,[X], O, [0,1,2,3,4]) :: category(X,O).

%%%%%%%%%%%%%%%% FashionMNIST
% Shoes : [5,7,9] = 0
% Tops : [0,6] = 1
% Overtop : [2,4] = 2
% Bag : [8] = 3
% Pants : [1] = 4
t(0.333) :: price(0,40); t(0.333) :: price(0,30); t(0.333) :: price(0,20).
t(0.5) :: price(1,10); t(0.5) :: price(1,2).
t(0.5) :: price(2,20); t(0.5) :: price(2,50).
price(3,60).
price(4,25).

%%%%%%%%%%%%%%%% DF
% % Shirt (tank vs general vs button down)
% t(0.25) :: price(0,5); t(0.5) :: price(0,10); t(0.25) :: price(0,15).
% % Pullover
% price(1,25).
% % Shorts
% price(2,20).
% % Casual pants
% price(3,25).
% % Normal pants (general vs chinos)
% t(0.666) :: price(4,40); t(0.333) :: price(4,45).
% % Dress, skirt, etc
% price(5,35).
% % Coat (In general vs anarok)
% t(0.666) :: price(6,40); t(0.333) :: price(6,50).

totalPrice(I1, I2, TP) :- 
    category(I1,C1),
    category(I2,C2),
    price(C1,P1),
    price(C2,P2),
    TP is P1 + P2.
